"""통합 챗봇 UI - 항공권, 기내식, FAQ를 하나의 챗봇에서 처리"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .agent import build_agent_executor, run_agent
from .faq_rag import answer_faq_question, get_supported_airlines
from .meal_rag import answer_meal_question
from .tts_helper import text_to_speech


def render_unified_chatbot() -> None:
    """통합 챗봇 인터페이스를 렌더링합니다."""
    
    _init_unified_chat()
    _render_sidebar()
    _render_chat_history()
    _handle_unified_input()


def _init_unified_chat() -> None:
    """챗봇 상태 초기화"""
    if "unified_messages" not in st.session_state:
        st.session_state.unified_messages = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! FRYND 통합 챗봇입니다. 😊\n\n"
                    "다음과 같은 질문에 답변드릴 수 있어요:\n\n"
                    "✈️ **항공권 검색** - 인천·김포·하네다·나리타 공항 간 항공권 정보\n\n"
                    "🍽️ **기내식 정보** - 항공사별 기내식 메뉴 및 제공 조건\n\n"
                    "❓ **FAQ** - 항공사별 자주 묻는 질문\n\n"
                    "무엇을 도와드릴까요?"
                ),
            }
        ]
    
    if "unified_langchain_history" not in st.session_state:
        st.session_state.unified_langchain_history = [
            SystemMessage(
                content=(
                    "너는 FRYND 통합 항공 상담 챗봇이다. 사용자의 질문 유형을 파악하여:\n"
                    "1. 항공권 검색/예약 관련 질문이면 항공권 검색 도구를 사용\n"
                    "2. 기내식 관련 질문이면 기내식 정보 제공\n"
                    "3. FAQ/일반 질문이면 FAQ 데이터베이스에서 답변\n"
                    "항상 한국어로 친절하게 답변하라."
                )
            )
        ]
    
    st.session_state.setdefault("unified_selected_airline", "전체")


def _render_sidebar() -> None:
    """사이드바에 항공사 선택 옵션 표시"""
    with st.sidebar:
        st.subheader("📋 FAQ 항공사 필터")
        airlines = ["전체"] + get_supported_airlines()
        selection = st.selectbox(
            "FAQ 검색 시 사용할 항공사",
            options=airlines,
            key="unified-airline-select"
        )
        st.session_state.unified_selected_airline = selection
        
        st.divider()
        st.caption(
            "💡 **팁**\n\n"
            "- 항공권 검색: '인천에서 도쿄 항공권 검색해줘'\n"
            "- 기내식 정보: '기내식 메뉴 알려줘'\n"
            "- FAQ: '수하물 규정 알려줘'"
        )


def _render_chat_history() -> None:
    """채팅 기록 렌더링"""
    for idx, message in enumerate(st.session_state.unified_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 항공권 테이블 표시
            table = message.get("table")
            if table is not None:
                df = pd.DataFrame(table)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 어시스턴트 메시지에 TTS 버튼 추가
            if message["role"] == "assistant":
                col1, col2 = st.columns([0.95, 0.05])
                with col2:
                    if st.button("🔊", key=f"tts_{idx}", help="음성으로 듣기"):
                        _play_tts(message["content"], idx)


def _handle_unified_input() -> None:
    """사용자 입력 처리 및 적절한 기능으로 라우팅"""
    user_input = st.chat_input("질문을 입력하세요...")
    if not user_input:
        return
    
    _append_message("user", user_input)
    history: List[BaseMessage] = st.session_state.unified_langchain_history
    
    with st.spinner("답변을 준비하고 있어요..."):
        try:
            # 질문 유형 분류
            query_type = _classify_query(user_input)
            
            if query_type == "flight":
                # 항공권 검색
                response, table = _handle_flight_query(user_input, history)
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=response))
                _append_message("assistant", response, table=table)
                
            elif query_type == "meal":
                # 기내식 정보
                response = answer_meal_question(user_input, history, top_k=3)
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=response))
                _append_message("assistant", response)
                
            else:  # faq
                # FAQ 검색
                airline_filter = st.session_state.get("unified_selected_airline")
                if airline_filter == "전체":
                    airline_filter = None
                response = answer_faq_question(user_input, history, top_k=4, airline=airline_filter)
                history.append(HumanMessage(content=user_input))
                history.append(AIMessage(content=response))
                _append_message("assistant", response)
                
        except Exception as exc:
            history.append(HumanMessage(content=user_input))
            error_msg = f"⚠️ 오류가 발생했습니다: {exc}"
            _append_message("assistant", error_msg)
    
    st.rerun()


def _classify_query(query: str) -> str:
    """질문 유형을 분류합니다 (flight, meal, faq)"""
    query_lower = query.lower()
    
    # 항공권 관련 키워드
    flight_keywords = [
        "항공권", "티켓", "예약", "검색", "운임", "가격",
        "인천", "김포", "하네다", "나리타", "icn", "gmp", "hnd", "nrt",
        "편도", "왕복", "비행", "출발", "도착", "언제"
    ]
    
    # 기내식 관련 키워드
    meal_keywords = [
        "기내식", "식사", "메뉴", "음식", "먹을", "기내 식사",
        "기내 메뉴", "제공", "특별식", "할랄", "채식"
    ]
    
    # 항공권 키워드 체크
    if any(keyword in query_lower for keyword in flight_keywords):
        return "flight"
    
    # 기내식 키워드 체크
    if any(keyword in query_lower for keyword in meal_keywords):
        return "meal"
    
    # 기본값은 FAQ
    return "faq"


def _handle_flight_query(query: str, history: List[BaseMessage]) -> tuple[str, List[dict] | None]:
    """항공권 검색 쿼리 처리"""
    executor = _get_agent_executor()
    return run_agent(executor, query, history)


def _append_message(role: str, content: str, table: List[Dict[str, str]] | None = None) -> None:
    """메시지를 채팅 기록에 추가"""
    message: Dict[str, Any] = {"role": role, "content": content}
    if table is not None:
        message["table"] = table
    st.session_state.unified_messages.append(message)


@st.cache_resource(show_spinner=False)
def _get_agent_executor():
    """에이전트 실행기를 캐싱하여 반환"""
    return build_agent_executor()


def _play_tts(text: str, message_idx: int) -> None:
    """텍스트를 음성으로 재생합니다."""
    try:
        # 마크다운 형식 제거 (간단한 정리)
        clean_text = _clean_text_for_tts(text)
        
        # TTS 생성
        with st.spinner("음성을 생성하고 있습니다..."):
            audio_bytes, audio_mime = text_to_speech(clean_text)
        
        # 오디오 재생
        st.audio(audio_bytes, format=audio_mime or "audio/wav", autoplay=True)
        
    except Exception as e:
        st.error(f"음성 생성 중 오류가 발생했습니다: {e}")


def _clean_text_for_tts(text: str) -> str:
    """TTS를 위해 텍스트를 정리합니다."""
    # 마크다운 링크 제거 [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 마크다운 강조 제거 (**text**, *text*, __text__, _text_)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # 마크다운 헤더 제거 (# text -> text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # 이모지는 유지하되 특수 기호 제거
    text = re.sub(r'[`~]', '', text)
    
    # 연속된 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    
    # 너무 긴 텍스트는 앞부분만 (최대 500자)
    if len(text) > 500:
        text = text[:500] + "... 자세한 내용은 화면을 참고해주세요."
    
    return text.strip()


__all__ = ["render_unified_chatbot"]
