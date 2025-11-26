"""Streamlit UI helpers for the in-flight meal RAG chatbot."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .meal_rag import answer_meal_question


def render_meal_chatbot() -> None:
    """Render the Streamlit chat interface for meal-related Q&A."""

    _init_chat_history()
    _render_chat_history()
    _handle_user_input()


def _init_chat_history() -> None:
    if "meal_messages" not in st.session_state:
        st.session_state["meal_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 항공사 기내식 관련 궁금한 점을 물어보세요.\n"
                    "내부 문서(RAG)를 참고해 정확한 정보를 찾아드릴게요."
                ),
            }
        ]

    if "meal_langchain_history" not in st.session_state:
        st.session_state["meal_langchain_history"] = [
            SystemMessage(
                content=(
                    "너는 항공사 기내식 관련 질문에 답하는 전문가다. 제공된 문서 내용만을 근거로 답변하고,"
                    " 문서에 없거나 불확실한 내용은 추측하지 말아야 한다."
                )
            )
        ]


def _render_chat_history() -> None:
    for message in st.session_state.meal_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _handle_user_input() -> None:
    user_input = st.chat_input("기내식 관련 질문을 입력하세요.", key="meal-chat-input")
    if not user_input:
        return

    _append_message("user", user_input)
    history: List[BaseMessage] = st.session_state["meal_langchain_history"]

    with st.spinner("기내식 자료를 검토하고 있어요..."):
        try:
            answer = answer_meal_question(user_input, history)
        except RuntimeError as exc:
            history.append(HumanMessage(content=user_input))
            _append_message("assistant", f"⚠️ {exc}")
        else:
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer))
            _append_message("assistant", answer)

    st.rerun()


def _append_message(role: str, content: str) -> None:
    message: Dict[str, Any] = {"role": role, "content": content}
    st.session_state.meal_messages.append(message)


__all__ = ["render_meal_chatbot"]
