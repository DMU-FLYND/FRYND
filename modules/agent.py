"""LangChain agent setup for the FRYND chatbot."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, List, Tuple

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from .amadeus import FlightSearchInput, flight_search_tool


def build_agent_executor() -> AgentExecutor:
    """Create the LangChain agent executor (without Streamlit caching)."""

    llm = _build_llm()
    flight_tool = _create_flight_tool()
    today_str = date.today().strftime("%Y-%m-%d")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 FRYND 항공권 상담 챗봇이야. 오늘 날짜는 {today}이며, 사용자가 자연어로 요청할 때"
                " 반드시 이 날짜를 기준으로 현재 시점을 판단해야 해. 항상 한국어로 친절하게 답변하고"
                " 필요한 경우 `flight_offer_lookup` 도구를 호출해 인천(ICN)·김포(GMP)·하네다(HND)·나리타(NRT)"
                " 공항 사이의 운임과 일정 정보를 찾아야 해. 도구에서 제공하는 정보 외에는 임의로 데이터를"
                " 만들어내면 안 돼.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    prompt = prompt.partial(today=today_str)
    agent = create_tool_calling_agent(llm, [flight_tool], prompt)
    return AgentExecutor(
        agent=agent,
        tools=[flight_tool],
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_agent(executor: AgentExecutor, user_input: str, history: List[BaseMessage]) -> Tuple[str, List[dict] | None]:
    """Invoke the agent and capture both text output and table payloads."""

    try:
        result = executor.invoke({"input": user_input, "chat_history": history})
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"LangChain 에이전트 실행 중 오류가 발생했습니다: {error}") from error

    output = result.get("output", "")
    table = extract_table_from_steps(result.get("intermediate_steps"))
    return output, table


def extract_table_from_steps(steps: Any) -> List[dict] | None:
    """Parse the intermediate tool responses to locate the latest table payload."""

    if not steps:
        return None

    extracted: List[dict] | None = None
    for _action, observation in steps:
        if not observation:
            continue

        payload: Any = observation
        if isinstance(observation, str):
            try:
                payload = json.loads(observation)
            except json.JSONDecodeError:
                continue

        if isinstance(payload, dict) and isinstance(payload.get("table"), list):
            extracted = payload["table"]

    return extracted


def _build_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Google Gemini API 자격증명이 없습니다. 환경 변수 GOOGLE_API_KEY를 설정해 주세요."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=2048,
    )


def _create_flight_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="flight_offer_lookup",
        description=(
            "Amadeus 항공권 조회 도구입니다. 인천(ICN), 김포(GMP), 하네다(HND), 나리타(NRT)"
            " 공항 사이의 편도 여정에 대해 출발일(YYYY-MM-DD)과 탑승객 수(1-9명)를 필수로 입력해야 합니다."
        ),
        func=flight_search_tool,
        args_schema=FlightSearchInput,
    )


__all__ = ["build_agent_executor", "run_agent", "extract_table_from_steps"]
