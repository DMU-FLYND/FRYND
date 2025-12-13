"""Streamlit entry point for the FRYND chatbot."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from modules.unified_chat_ui import render_unified_chatbot
from modules.realtime_voice import render_realtime_voice_ui

load_dotenv()


def main() -> None:
    st.set_page_config(page_title="FRYND", page_icon="✈️", layout="wide")
    st.title("FRYND✈️ 통합 항공 상담 챗봇")
    st.caption("항공권 검색, 기내식 정보, FAQ를 한 곳에서! 🚀")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["💬 텍스트 챗봇", "🎙️ 음성 대화 (Realtime API)"])
    
    with tab1:
        render_unified_chatbot()
    
    with tab2:
        render_realtime_voice_ui()


if __name__ == "__main__":
    main()
