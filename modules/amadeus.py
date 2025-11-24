"""Amadeus flight search utilities and LangChain tool entry point."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from amadeus import Client, ResponseError
from pydantic import BaseModel, Field

from .constants import AIRPORTS


class FlightSearchInput(BaseModel):
    """Schema that LangChain uses for tool-call validation."""

    origin: str = Field(
        ...,
        description="출발 공항 IATA 코드 (ICN, GMP, HND, NRT 중 하나).",
    )
    destination: str = Field(
        ...,
        description="도착 공항 IATA 코드 (ICN, GMP, HND, NRT 중 하나).",
    )
    departure_date: str = Field(
        ...,
        description="출발일 (YYYY-MM-DD).",
    )
    adults: int = Field(
        1,
        ge=1,
        le=9,
        description="탑승객 수 (1-9명).",
    )


def flight_search_tool(origin: str, destination: str, departure_date: str, adults: int = 1) -> str:
    """LangChain tool wrapper for the Amadeus flight search."""

    try:
        origin_code = _validate_airport_code(origin, "출발")
        destination_code = _validate_airport_code(destination, "도착")
        if origin_code == destination_code:
            raise ValueError("출발 공항과 도착 공항은 서로 달라야 합니다.")
        if not (1 <= adults <= 9):
            raise ValueError("탑승객 수는 1명 이상 9명 이하로 입력해 주세요.")

        departure = _parse_departure_date(departure_date)
        offers = _search_flights(origin_code, destination_code, departure, adults)
        summary, table = _prepare_offer_response(
            offers,
            origin_code,
            destination_code,
            departure,
            adults,
        )
        payload = {"summary": summary, "table": table}
    except ValueError as error:
        payload = {
            "summary": str(error),
            "table": None,
            "error": True,
        }
    except Exception as error:  # noqa: BLE001
        payload = {
            "summary": f"Amadeus API 호출 중 오류가 발생했습니다: {error}",
            "table": None,
            "error": True,
        }

    return json.dumps(payload, ensure_ascii=False)


def _validate_airport_code(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} 공항 코드를 입력해 주세요.")

    code = value.strip().upper()
    if code not in AIRPORTS:
        allowed = ", ".join(AIRPORTS.keys())
        raise ValueError(f"{field_name} 공항은 {allowed} 중 하나여야 합니다.")

    return code


def _parse_departure_date(value: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError as error:  # noqa: BLE001
        raise ValueError("출발일은 YYYY-MM-DD 형식으로 입력해 주세요.") from error


@lru_cache(maxsize=8)
def _cached_amadeus_client(client_id: str, client_secret: str, hostname: str) -> Client:
    return Client(client_id=client_id, client_secret=client_secret, hostname=hostname)


def _get_amadeus_client() -> Client:
    client_id = os.getenv("AMADEUS_CLIENT_ID")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
    hostname = os.getenv("AMADEUS_HOSTNAME", "test")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Amadeus API 자격증명이 없습니다. 환경 변수 AMADEUS_CLIENT_ID / AMADEUS_CLIENT_SECRET를 설정해 주세요."
        )

    return _cached_amadeus_client(client_id, client_secret, hostname)


def _search_flights(origin: str, destination: str, departure_date: date, adults: int) -> List[Dict[str, Any]]:
    client = _get_amadeus_client()
    try:
        response = client.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date.strftime("%Y-%m-%d"),
            adults=adults,
            currencyCode="KRW",
            max=5,
        )
    except ResponseError as error:
        parsed = _format_amadeus_error(error)
        raise RuntimeError(parsed) from error

    return response.data or []


def _prepare_offer_response(
    offers: List[Dict[str, Any]],
    origin: str,
    destination: str,
    departure_date: date,
    adults: int,
) -> Tuple[str, List[Dict[str, str]] | None]:
    header = (
        f"{AIRPORTS[origin]} → {AIRPORTS[destination]} ({departure_date.strftime('%Y-%m-%d')}, {adults}명)\n"
        "상위 5개 옵션을 보여드릴게요."
    )

    if not offers:
        return header + "\n\n조회 가능한 항공편이 없습니다. 날짜나 인원을 조정해 보세요.", None

    table = _build_offer_table(offers)
    message = header + "\n\n아래 표에서 세부 정보를 확인해 주세요.\n\n※ 실제 운임과 좌석은 조회 시점에 따라 달라질 수 있습니다."
    return message, table


def _build_offer_table(offers: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for idx, offer in enumerate(offers, start=1):
        rows.append(_offer_to_row(idx, offer))
    return rows


def _offer_to_row(index: int, offer: Dict[str, Any]) -> Dict[str, str]:
    price = offer.get("price", {})
    itineraries = offer.get("itineraries", [])
    itinerary = itineraries[0] if itineraries else {}
    segments = itinerary.get("segments", [])

    if not segments:
        return {
            "옵션": f"{index}",
            "여정": "정보 없음",
            "출발": "-",
            "도착": "-",
            "소요 시간": "-",
            "항공사": "-",
            "항공편": "-",
            "요금": _format_price(price),
        }

    first_segment = segments[0]
    last_segment = segments[-1]
    departure_time = _format_iso_timestamp(first_segment.get("departure", {}).get("at", ""))
    arrival_time = _format_iso_timestamp(last_segment.get("arrival", {}).get("at", ""))
    stop_count = len(segments) - 1
    stop_text = "직항" if stop_count == 0 else f"경유 {stop_count}회"
    duration = _humanize_duration(itinerary.get("duration", ""))
    airline = first_segment.get("carrierCode", "알수없음")
    flight_numbers = ", ".join(
        f"{seg.get('carrierCode', '')}{seg.get('number', '')}".strip()
        for seg in segments
        if seg.get("number")
    )

    return {
        "옵션": f"{index}",
        "여정": stop_text,
        "출발": departure_time,
        "도착": arrival_time,
        "소요 시간": duration,
        "항공사": airline,
        "항공편": flight_numbers or "편명 정보 없음",
        "요금": _format_price(price),
    }


def _format_amadeus_error(error: ResponseError) -> str:
    response = getattr(error, "response", None)
    body = getattr(response, "body", None)

    details: List[str] = []
    if body:
        try:
            payload = json.loads(body)
            if isinstance(payload, list):
                errors = payload
            elif isinstance(payload, dict):
                errors = payload.get("errors", [])
            else:
                errors = []
            for item in errors:
                if isinstance(item, dict):
                    title = item.get("title")
                    detail = item.get("detail") or item.get("description")
                    parameter = (item.get("source") or {}).get("parameter")
                    pieces = [piece for piece in [title, detail, parameter] if piece]
                    if pieces:
                        details.append(" - ".join(pieces))
        except (TypeError, json.JSONDecodeError):
            pass

    if details:
        return "Amadeus API 오류: " + "; ".join(details)

    return f"Amadeus API 오류: HTTP {getattr(response, 'status_code', '400')}"


def _format_price(price: Dict[str, Any]) -> str:
    grand_total = price.get("grandTotal")
    currency = price.get("currency", "")
    if grand_total and currency:
        return f"{grand_total} {currency}"
    if grand_total:
        return str(grand_total)
    return "요금 정보 없음"


def _format_iso_timestamp(value: str) -> str:
    if not value:
        return "시간 정보 없음"

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return value

    return parsed.strftime("%Y-%m-%d %H:%M")


def _humanize_duration(duration: str) -> str:
    if not duration:
        return "소요 시간 정보 없음"

    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration)
    if not match:
        return duration

    hours, minutes = match.groups()
    parts = []
    if hours:
        parts.append(f"{int(hours)}시간")
    if minutes:
        parts.append(f"{int(minutes)}분")

    return " ".join(parts) if parts else duration


__all__ = [
    "FlightSearchInput",
    "flight_search_tool",
]
