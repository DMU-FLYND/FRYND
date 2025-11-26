"""Static constants for the FRYND chatbot."""

from __future__ import annotations

from typing import Dict

AIRPORTS: Dict[str, str] = {
    "ICN": "인천국제공항 (ICN) · Seoul",
    "GMP": "김포국제공항 (GMP) · Seoul",
    "HND": "도쿄국제공항 (HND) · Tokyo",
    "NRT": "나리타국제공항 (NRT) · Tokyo",
}

__all__ = ["AIRPORTS"]
