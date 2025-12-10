"""Static constants for the FRYND chatbot."""

from __future__ import annotations

from typing import Dict

AIRPORTS: Dict[str, str] = {
    "ICN": "인천국제공항 (ICN) · Seoul",
    "GMP": "김포국제공항 (GMP) · Seoul",
    "HND": "도쿄국제공항 (HND) · Tokyo",
    "NRT": "나리타국제공항 (NRT) · Tokyo",
}

AIRLINES: Dict[str, str] = {
    "KE": "대한항공",
    "OZ": "아시아나항공",
    "LJ": "진에어",
    "7C": "제주항공",
    "TW": "티웨이항공",
    "BX": "에어부산",
    "ZE": "이스타항공",
    "RS": "에어서울",
    "4V": "에어프레미아",
    "JL": "일본항공",
    "NH": "전일본공수",
    "MM": "피치항공",
    "GK": "젯스타 재팬",
    "BC": "스카이마크",
    "LQ": "솔라시도 에어",
    "AD": "에어 두",
}

__all__ = ["AIRPORTS", "AIRLINES"]
