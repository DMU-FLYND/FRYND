"""Streamlit entry point for the FRYND chatbot."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from modules.chat_ui import render_chatbot

load_dotenv()


def main() -> None:
    st.set_page_config(page_title="FRYND 항공권 챗봇", page_icon="✈️", layout="centered")
    st.title("FRYND 항공권 챗봇 ✈️")
    st.caption("LangChain + Gemini 2.5 Flash + Amadeus API로 구현한 항공권 상담 챗봇")
    st.info(
        "인천(ICN)·김포(GMP)·하네다(HND)·나리타(NRT) 공항 사이의 항공권 정보를 자연어로 문의하세요.\n"
        "필요 시 챗봇이 Amadeus API 함수를 호출해 최신 운임을 찾아드립니다."
    )

    render_chatbot()


if __name__ == "__main__":
    main()
