"""Streamlit UI helpers for the FAQ RAG chatbot."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .faq_rag import answer_faq_question, get_supported_airlines


def render_faq() -> None:
    """Render the Streamlit chat interface for FAQ Q&A."""

    _init_chat_state()
    _render_airline_filter()
    _render_chat_history()
    _handle_user_input()


def _init_chat_state() -> None:
    if "faq_messages" not in st.session_state:
        st.session_state["faq_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 항공사 FAQ를 알려드릴게요.\n"
                    "아래 질문을 입력하고 필요한 경우 항공사를 선택하세요."
                ),
            }
        ]

    if "faq_langchain_history" not in st.session_state:
        st.session_state["faq_langchain_history"] = [
            SystemMessage(
                content=(
                    "너는 항공사 FAQ에 답하는 전문가다. 제공된 FAQ 문서를 근거로 답하고, "
                    "문서에 없는 내용은 추측하지 않는다."
                )
            )
        ]

    st.session_state.setdefault("faq_airline", "전체")


def _render_airline_filter() -> None:
    airlines = ["전체"] + get_supported_airlines()
    selection = st.selectbox("검색할 항공사를 선택하세요.", options=airlines, key="faq-airline-select")
    st.session_state["faq_airline"] = selection


def _render_chat_history() -> None:
    for message in st.session_state.faq_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_user_input() -> None:
    user_input = st.chat_input("항공사 FAQ를 물어보세요.", key="faq-chat-input")
    if not user_input:
        return

    _append_message("user", user_input)
    history: List[BaseMessage] = st.session_state["faq_langchain_history"]

    airline_choice = st.session_state.get("faq_airline") or "전체"
    airline_filter = None if airline_choice == "전체" else airline_choice

    with st.spinner("FAQ 자료를 검토하고 있어요..."):
        try:
            answer = answer_faq_question(user_input, history, airline=airline_filter)
        except RuntimeError as exc:
            history.append(HumanMessage(content=user_input))
            _append_message("assistant", f"⚠️ {exc}")
        except Exception as exc:  # noqa: BLE001
            history.append(HumanMessage(content=user_input))
            _append_message("assistant", f"⚠️ FAQ 처리 중 오류가 발생했어요: {exc}")
        else:
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer))
            _append_message("assistant", answer)

    st.rerun()


def _append_message(role: str, content: str) -> None:
    message: Dict[str, Any] = {"role": role, "content": content}
    st.session_state.faq_messages.append(message)


__all__ = ["render_faq"]
