"""Wrapper around EnhancedMaterialDatabase (supplier + recommendations)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .._legacy_imports import anomaly_mod as _canon
from ..models import FatigueParameters, MaterialRecommendation


_DB = _canon.EnhancedMaterialDatabase()


def get_material_params(material_name: str) -> Optional[FatigueParameters]:
    """Global accessor used across engines/adapters."""
    return _DB.get_material(material_name)


class MaterialEngine:
    """
    Surfaces material-change + supplier recommendations. Called by the
    Action Agent only when RUL is low (per rule #6 in the build prompt).
    """

    def __init__(self) -> None:
        self._db = _DB

    def recommend(self, current_material: str) -> Dict[str, Any]:
        best: Optional[MaterialRecommendation] = self._db.get_best_recommendation(current_material)
        alternatives: List[MaterialRecommendation] = self._db.get_material_recommendations(current_material)
        return {
            "current_material": current_material,
            "best_recommendation": _rec_to_dict(best) if best else None,
            "alternatives": [_rec_to_dict(r) for r in alternatives],
            "supplier_of_current": (
                self._db.get_material(current_material).supplier
                if self._db.get_material(current_material)
                else None
            ),
        }


def _rec_to_dict(rec: MaterialRecommendation) -> Dict[str, Any]:
    return {
        "current_material": rec.current_material,
        "recommended_material": rec.recommended_material,
        "supplier": rec.supplier,
        "part_number": rec.part_number,
        "reason": rec.reason,
        "expected_improvement": rec.expected_improvement,
        "cost_impact": rec.cost_impact,
        "implementation_days": rec.implementation_days,
        "risk_level": rec.risk_level,
        "recommendation_type": rec.recommendation_type.value,
        "confidence_score": rec.confidence_score,
    }
