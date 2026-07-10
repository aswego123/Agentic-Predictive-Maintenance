"""IBM Maximo stub adapter."""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MaximoAdapter:
    name = "ibm_maximo"

    def create_work_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        work_order_id = f"MAX-WO-{uuid.uuid4().hex[:10].upper()}"
        wo = {
            "provider": self.name,
            "work_order_id": work_order_id,
            "status": "created_stub",
            "payload": payload,
        }
        logger.info("[Maximo stub] Would create work order: %s", wo)
        return wo
