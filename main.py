"""Streamlit entry point for the FRYND chatbot."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from modules.chat_ui import render_chatbot
from modules.meal_chat_ui import render_meal_chatbot

load_dotenv()


def main() -> None:
    st.set_page_config(page_title="FRYND", page_icon="✈️", layout="centered")
    st.title("FRYND✈️")
    st.caption("항공 여정 및 기내식 안내 챗봇")

    mode = st.radio(
        "원하는 상담 모드를 선택하세요.",
        ("항공권 상담", "기내식 Q&A"),
        horizontal=True,
    )

    if mode == "항공권 상담":
        st.info(
            "인천(ICN)·김포(GMP)·하네다(HND)·나리타(NRT) 공항 사이의 항공권 정보를 챗봇으로 문의하세요.\n"
            "필요 시 챗봇이 Amadeus API 함수를 호출해 최신 운임을 찾아드립니다."
        )
        render_chatbot()
    else:
        st.info(
            "항공사 기내식 관련 질문을 하면 내부 문서(RAG)를 바탕으로 답변을 제공합니다.\n"
            "메뉴, 제공 조건, 대상 항공사 등을 구체적으로 물어보세요."
        )
        render_meal_chatbot()


if __name__ == "__main__":
    main()
