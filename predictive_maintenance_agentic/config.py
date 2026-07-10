"""
Central configuration for the agentic predictive-maintenance system.

Everything provider-specific (LLM vendor, thresholds, iteration caps) is
centralized here so the graph, agents, and API layer stay portable
between demo environments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# Optional: load a repo-local `.env` if python-dotenv is installed.
# Keeps secrets out of the shell history / process listing.
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# ============================================================
# LLM PROVIDER SELECTION
# ============================================================
# Set EIX_LLM_PROVIDER to one of:
#   "anthropic", "openai", "azure", "azure_apim", "none".
# "none" disables LLM calls entirely — agents fall back to their
# deterministic rule-based reasoning path. This keeps the graph
# runnable in offline / hackathon-demo environments.

LLM_PROVIDER: str = os.getenv("EIX_LLM_PROVIDER", "none").lower()
LLM_MODEL: str = os.getenv(
    "EIX_LLM_MODEL",
    {
        "anthropic": "claude-3-5-sonnet-latest",
        "openai": "gpt-4o-mini",
        "azure": "gpt-4o-mini",
        "azure_apim": "gpt-5-codex",
        "none": "",
    }.get(LLM_PROVIDER, ""),
)
LLM_TEMPERATURE: float = float(os.getenv("EIX_LLM_TEMPERATURE", "0.1"))


def _build_llm(provider: str, model: str, temp: float):
    """Internal factory shared by get_llm() and get_judge_llm()."""
    if provider == "none" or not provider:
        return None

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "provider=anthropic but langchain-anthropic is not installed"
            ) from exc
        return ChatAnthropic(model=model, temperature=temp)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "provider=openai but langchain-openai is not installed"
            ) from exc
        return ChatOpenAI(model=model, temperature=temp)

    if provider == "google":
        # Google Gemini (Generative Language API) — keyed via
        # EIX_GOOGLE_API_KEY. Used for the Judge LLM by default so
        # ReAct reasoning is independent of the primary Azure LLM.
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "provider=google but langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            ) from exc
        api_key = os.getenv("EIX_GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("provider=google but EIX_GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-pro",
            temperature=temp,
            google_api_key=api_key,
        )

    if provider in {"azure", "azure_apim"}:
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "provider=azure requires langchain-openai to be installed"
            ) from exc
        endpoint    = os.getenv("EIX_AZURE_ENDPOINT")
        base_url    = os.getenv("EIX_AZURE_BASE_URL")
        api_key     = os.getenv("EIX_AZURE_API_KEY")
        api_version = os.getenv("EIX_AZURE_API_VERSION", "2024-06-01")
        deployment  = os.getenv("EIX_AZURE_DEPLOYMENT", model)

        missing = [
            name for name, val in [
                ("EIX_AZURE_ENDPOINT or EIX_AZURE_BASE_URL", endpoint or base_url),
                ("EIX_AZURE_API_KEY", api_key),
                ("EIX_AZURE_DEPLOYMENT", deployment),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"provider={provider} but missing env vars: {missing}."
            )

        if base_url:
            from langchain_openai import ChatOpenAI
            full_base = f"{base_url.rstrip('/')}/deployments/{deployment}"
            return ChatOpenAI(
                model=deployment,
                temperature=temp,
                base_url=full_base,
                api_key=api_key,
                default_headers={
                    "Ocp-Apim-Subscription-Key": api_key,
                    "api-key": api_key,
                },
                default_query={"api-version": api_version},
            )

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=temp,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def get_llm(temperature: Optional[float] = None):
    """
    Return the primary LangChain chat model (used by physics/ML
    narration + fallback Judge path), or None when disabled.
    """
    temp = LLM_TEMPERATURE if temperature is None else temperature
    return _build_llm(LLM_PROVIDER, LLM_MODEL, temp)


def get_judge_llm(temperature: Optional[float] = None):
    """
    Return the Judge Agent's LLM. Independently configurable via
    EIX_JUDGE_LLM_PROVIDER / EIX_JUDGE_LLM_MODEL / provider-specific
    keys. Defaults to `get_llm()` if EIX_JUDGE_LLM_PROVIDER is unset
    or "inherit".
    """
    provider = os.getenv("EIX_JUDGE_LLM_PROVIDER", "inherit").lower()
    if provider in {"", "inherit"}:
        return get_llm(temperature)
    model = os.getenv("EIX_JUDGE_LLM_MODEL") or LLM_MODEL
    temp = LLM_TEMPERATURE if temperature is None else temperature
    return _build_llm(provider, model, temp)


# ============================================================
# GRAPH BEHAVIOR (matches sections 4-5 of the build prompt)
# ============================================================

@dataclass
class GraphLimits:
    """Iteration caps enforced by the LangGraph state machine."""
    # Physics ⇄ ML negotiation
    max_negotiation_rounds: int = 2
    # Judge → Calibration → Engineer → re-simulate loop
    max_resimulation_rounds: int = 5


@dataclass
class Thresholds:
    """Numeric thresholds used by the anomaly gate + judge."""
    # Judge Agent: divergence in predicted stress values (MPa) beyond
    # which physics and ML are treated as disagreeing.
    stress_divergence_mpa: float = 25.0
    # Fractional divergence in RUL (hours); e.g. 0.20 => 20% delta.
    rul_divergence_fraction: float = 0.20
    # Confidence score below which Judge routes to Calibration even
    # without absolute divergence.
    min_confidence_score: float = 0.65
    # RUL below this many hours triggers material-change surfacing in
    # the Action Agent (per rule #6 in the build prompt).
    low_rul_hours: float = 500.0


@dataclass
class MemoryConfig:
    """Where to persist LangGraph state + Fleet Memory."""
    checkpoint_path: str = field(
        default_factory=lambda: os.getenv(
            "EIX_CHECKPOINT_PATH", "./sqlite-data/eix_checkpoints.sqlite"
        )
    )
    fleet_memory_path: str = field(
        default_factory=lambda: os.getenv(
            "EIX_FLEET_MEMORY_PATH", "./sqlite-data/eix_fleet_memory.sqlite"
        )
    )


LIMITS = GraphLimits()
THRESHOLDS = Thresholds()
MEMORY = MemoryConfig()
