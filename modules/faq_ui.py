"""FAQ UI module using Streamlit."""

import streamlit as st
from modules.faq import get_faq_response  # faq.py에서 직접 import


def render_faq() -> None:
    """Render FAQ chatbot interface."""
    
    # Initialize chat history
    if "faq_messages" not in st.session_state:
        st.session_state.faq_messages = []
    
    # Display chat history
    for message in st.session_state.faq_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("FAQ 질문을 입력하세요 (예: 제주항공의 국내선 수하물 허용량은?)"):
        # Add user message
        st.session_state.faq_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = get_faq_response(prompt)
                st.markdown(response)
        
        st.session_state.faq_messages.append({"role": "assistant", "content": response})