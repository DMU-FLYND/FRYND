"""Streamlit UI helpers for the FRYND chatbot."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .agent import build_agent_executor, run_agent


def render_chatbot() -> None:
    """Render the Streamlit chat interface and handle user prompts."""

    _init_chat_history()
    _render_chat_history()
    _handle_user_input()


def _init_chat_history() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 인천·김포·하네다·나리타 공항 사이의 항공권을 찾아드릴게요.\n"
                    "희망하는 여정을 자연어로 알려주시면 Amadeus 데이터를 기반으로 도와드립니다."
                ),
            }
        ]

    st.session_state.setdefault("langchain_history", [])


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            table = message.get("table")
            if table is not None:
                df = pd.DataFrame(table)
                st.dataframe(df, use_container_width=True, hide_index=True)


def _handle_user_input() -> None:
    user_input = st.chat_input("원하시는 여정을 자연어로 입력하세요.")
    if not user_input:
        return

    _append_message("user", user_input)
    history: List[BaseMessage] = st.session_state.setdefault("langchain_history", [])

    executor = _get_agent_executor()
    with st.spinner("FRYND가 답변을 준비하고 있어요..."):
        try:
            assistant_reply, table = run_agent(executor, user_input, history)
        except RuntimeError as exc:
            history.append(HumanMessage(content=user_input))
            _append_message("assistant", f"⚠️ {exc}")
        else:
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=assistant_reply))
            _append_message("assistant", assistant_reply, table=table)

    st.rerun()


def _append_message(role: str, content: str, table: List[Dict[str, str]] | None = None) -> None:
    message: Dict[str, Any] = {"role": role, "content": content}
    if table is not None:
        message["table"] = table
    st.session_state.messages.append(message)


@st.cache_resource(show_spinner=False)
def _get_agent_executor():
    return build_agent_executor()


__all__ = ["render_chatbot"]
