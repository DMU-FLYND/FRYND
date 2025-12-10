"""Streamlit entry point for the FRYND chatbot."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from modules.unified_chat_ui import render_unified_chatbot

load_dotenv()


def main() -> None:
    st.set_page_config(page_title="FRYND", page_icon="✈️", layout="wide")
    st.title("FRYND✈️ 통합 항공 상담 챗봇")
    st.caption("항공권 검색, 기내식 정보, FAQ를 한 곳에서! 🚀")
    
    render_unified_chatbot()


if __name__ == "__main__":
    main()
