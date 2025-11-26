import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb
import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from chatbot_func import * 
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv('key.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
query_augmentation_prompt = query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 질문을 재작성하는 도우미이다. "
            "대화 내용을 참고해 사용자의 마지막 질문의 의미를 명확한 한 문장으로 재작성하라. "
            "대명사(이, 저, 그 등)은 구체적 명사로 바꿔라."
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "재작성할 질문: {query}"
        )
    ]
)
 # 대화 내용을 보고 사용자의 질문을 명확한 한 문장으로 재작성하는 프롬프트

st.title('항공사 기내식 Q & A')

# 2. 처음 접속 시 챗봇 역할 규칙을 세팅

if 'messages' not in st.session_state: 
    st.session_state['messages'] = [
        SystemMessage("""
            너는 지금부터 항공사별 기내식 관련 질문에 답하는 전문가야.\n 
            다음 문서를 기반으로 정확하게 응답해줘.""")
    ] 


# 3. 세션에 저장된 이전 대화 메시지를 Streamlit 채팅 UI에 다시 출력
for msg in st.session_state.messages:
    if msg.content:
        if isinstance(msg, SystemMessage):
            st.chat_message('system').write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message('assistant').write(msg.content)
        elif isinstance(msg, HumanMessage):
            st.chat_message('user').write(msg.content)

# 4. 

lc_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
query_augmentation_chain = query_augmentation_prompt | lc_model | StrOutputParser()
chroma_client = chromadb.PersistentClient(path="D:\chatbot_project\chroma_db")

db = chroma_client.get_collection(
    name='chroma_db', 
    embedding_function=GeminiEmbeddingFunction()
)

if query := st.chat_input(): # 사용자 입력 받기
    st.chat_message('user').write(query) # 사용자 입력을 화면에 출력
    st.session_state.messages.append(HumanMessage(query))
    
    print('user\t',query)
    passage = get_relevant_passage(query, db, 3)

    augmented_query = query_augmentation_chain.invoke({
        'messages' : st.session_state['messages'],
        'query' : make_prompt(query, passage)
    })
    print('augmented_query\t', augmented_query)

    prompt_text = f"""
    아래는 관련 문서입니다. 문서 내용만을 근거로 답해라.

    [관련 문서]
    {passage}

    [사용자 질문]
    {augmented_query}

    답변:
    """

    with st.spinner(f'ai가 답변 준비중 {augmented_query}'):
        response = model.generate_content(prompt_text)
        answer = response.text
        result = st.chat_message('assistant').write(answer)
    
    st.session_state['messages'].append(AIMessage(answer))
    print("st.session_state['messages']", st.session_state['messages'])