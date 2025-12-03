import os
import json
import chromadb
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


load_dotenv(find_dotenv(), override=True)

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

# 항공사별 JSON 파일 매핑
AIRLINE_FILES = {
    "진에어": "jinair.json",
    "에어부산": "airbusan.json",
    "티웨이": "tway.json",
    "제주": "jeju.json",
    "에어프레미아": "airpremia.json"
}


# JSON 파일 로드 함수
def load_faq(airline_name):
    airline_name = airline_name.strip()

    if airline_name not in AIRLINE_FILES:
        raise ValueError(f"지원하지 않는 항공사입니다: {airline_name}")

    # 프로젝트 루트에서 faq/after_data 경로로 변경
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, "faq", "after_data", AIRLINE_FILES[airline_name])
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    return faq_data


# OpenAI Embedding을 사용한 ChromaDB 초기화
from chromadb.utils import embedding_functions

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPEN_API_KEY"),
    model_name="text-embedding-3-small"
)

# ChromaDB 절대 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
chroma_db_path = os.path.join(project_root, "data", "chroma_faq_db")

# 폴더가 없으면 생성
os.makedirs(chroma_db_path, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=chroma_db_path)
collection = chroma_client.get_or_create_collection(
    name="airline_faq",
    metadata={"hnsw:space": "cosine"},
    embedding_function=openai_ef
)


# FAQ 데이터를 벡터 DB에 삽입
def insert_faqs(airline, faq_data):
    documents = []
    metadatas = []
    ids = []

    for idx, item in enumerate(faq_data["faqs"]):
        content = item["question"] + " " + item["answer"]

        documents.append(content)
        metadatas.append({"airline": airline})
        ids.append(f"{airline}_{idx}")

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


def is_airline_mentioned(question):
    """질문에 항공사가 명시적으로 언급되었는지 확인"""
    airline_keywords = ["진에어", "에어부산", "티웨이", "제주", "에어프레미아"]
    question_lower = question.lower()
    
    for keyword in airline_keywords:
        if keyword in question_lower:
            return True
    return False


# GPT로 질문에서 항공사 추출 (여러 항공사 가능)
def extract_airlines(question, conversation_history, last_airline=None):
    # 항공사가 명시적으로 언급된 경우 대화 히스토리 무시
    if is_airline_mentioned(question):
        conversation_history = []  # 히스토리 초기화로 이전 맥락 제거
    
    history_text = "\n".join([f"사용자: {h['user']}\n봇: {h['bot']}" for h in conversation_history[-3:]])
    
    prompt = f"""
사용자의 질문에서 항공사를 파악하세요.
지원 항공사: 진에어, 에어부산, 티웨이, 제주항공, 에어프레미아

최근 대화 내역:
{history_text if history_text else "없음"}

현재 질문: {question}
이전 항공사: {last_airline if last_airline else "없음"}

**규칙**:
- 질문에 항공사 이름이 있으면 그 항공사만 답변
- 없으면 이전 항공사 유지
- 여러 항공사 비교 시 모두 답변 (쉼표 구분)

항공사명만 답변:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    airline_text = response.choices[0].message.content.strip()
    airlines = [a.strip() for a in airline_text.split(",")]
    valid_airlines = [a for a in airlines if a in AIRLINE_FILES]
    
    return valid_airlines if valid_airlines else ([last_airline] if last_airline else None)


# 질문에서 핵심 키워드 추출 함수 추가
def extract_keywords(question):
    prompt = f"""
다음 질문에서 핵심 키워드만 추출하세요.
예: "진에어 수하물 무게 제한이 어떻게 되나요?" -> "수하물, 무게, 제한"

질문: {question}

핵심 키워드만 쉼표로 구분하여 답변:
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()


# 검색 + gpt-4o-mini 답변 생성 (대화 히스토리 포함)
def generate_answer(question, airline, conversation_history):
    # 키워드 추출
    keywords = extract_keywords(question)
    # print(f"추출된 키워드: {keywords}")
    
    # 키워드로 검색 쿼리 생성
    keyword_list = [k.strip() for k in keywords.split(",")]
    search_query = question + " " + " ".join(keyword_list)  # 질문 + 키워드 결합
    
    # 키워드 개수에 따라 검색 결과 수 조정
    num_results = max(5, len(keyword_list) * 3)  # 최소 5개, 키워드당 3개씩
    num_results = min(num_results, 15)  # 최대 15개로 제한
    
    # print(f"검색 쿼리: {search_query}")
    # print(f"요청 FAQ 개수: {num_results}")
    
    # 항공사 필터링 검색
    results = collection.query(
        query_texts=[search_query],  # 키워드 포함 검색
        n_results=num_results,
        where={"airline": airline}
    )

    retrieved_docs = results["documents"][0]
    retrieved_distances = results["distances"][0] if "distances" in results else []
    
    # 유사도 점수로 관련성 필터링 (코사인 거리 0.7 이하만)
    filtered_docs = []
    for i, doc in enumerate(retrieved_docs):
        if i < len(retrieved_distances):
            distance = retrieved_distances[i]
            if distance < 0.7:  # 유사도가 높은 것만
                filtered_docs.append(doc)
            #     print(f"  ✓ FAQ {i+1} (유사도: {1-distance:.2f})")
            # else:
            #     print(f"  ✗ FAQ {i+1} (유사도 낮음: {1-distance:.2f})")
        else:
            filtered_docs.append(doc)
    
    # 검색된 문서가 관련있는지 확인
    if not filtered_docs or len(filtered_docs) == 0:
        return f"죄송합니다. {airline} 항공사의 '{keywords}' 관련 정보를 찾을 수 없습니다."
    
    # print(f"📚 필터링된 FAQ 개수: {len(filtered_docs)}")
    
    # 대화 히스토리를 메시지 형태로 변환
    messages = [
        {"role": "system", "content": f"당신은 {airline} 항공사 고객센터 상담원입니다. 제공된 FAQ 정보를 정확히 참고하여 답변하세요."}
    ]
    
    # 최근 3턴의 대화 추가
    for hist in conversation_history[-3:]:
        messages.append({"role": "user", "content": hist["user"]})
        messages.append({"role": "assistant", "content": hist["bot"]})
    
    # 현재 질문과 FAQ 정보
    faq_context = "\n\n".join([f"[FAQ {i+1}]\n{doc}" for i, doc in enumerate(filtered_docs)])
    
    current_prompt = f"""
아래는 {airline} 항공사의 관련 FAQ 내용입니다:

{faq_context}

추출된 핵심 키워드: {keywords}

사용자 질문:
{question}

**답변 가이드**:
1. FAQ에 구체적인 정보(금액, 기간, 절차 등)가 있으면 그대로 정확히 안내
2. 여러 FAQ에 분산된 정보는 종합하여 완전한 답변 제공
3. FAQ에 "홈페이지 참고"만 있으면 일반 정보와 함께 안내
4. 항공사 고객센터 상담원처럼 정중하고 친절하게 답변
5. 이전 대화 맥락을 고려하여 자연스럽게 답변
6. 300자 이내로 간결하게 답변
7. FAQ 정보에 링크가 있으면 답변 하단에 함께 표시
8. 답변 시 "**" 같은 표시 사용 금지
9. FAQ에 정보가 없으면 "죄송합니다, 관련 정보를 찾을 수 없습니다."라고 답변과 함께 각 항공사 고객센터 연락처 안내
"""
    
    messages.append({"role": "user", "content": current_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content


# Streamlit용 래퍼 함수 추가
def get_faq_response(question):
    """Streamlit UI에서 호출할 함수"""
    
    # 세션 히스토리 가져오기 (없으면 빈 리스트)
    import streamlit as st
    if "faq_conversation_history" not in st.session_state:
        st.session_state.faq_conversation_history = []
    if "faq_last_airline" not in st.session_state:
        st.session_state.faq_last_airline = None
    
    conversation_history = st.session_state.faq_conversation_history
    last_airline = st.session_state.faq_last_airline
    
    # 항공사 추출
    airlines = extract_airlines(question, conversation_history, last_airline)
    
    if not airlines:
        return "항공사를 파악할 수 없습니다. 항공사 이름(진에어, 에어부산, 티웨이, 제주, 에어프레미아)을 포함해주세요."
    
    airline = airlines[-1]
    
    # 답변 생성
    answer = generate_answer(question, airline, conversation_history)
    
    # 히스토리 업데이트
    st.session_state.faq_conversation_history.append({
        "user": question,
        "bot": answer,
        "airline": airline
    })
    st.session_state.faq_last_airline = airline
    
    return answer


# 메인 실행부
if __name__ == "__main__":
    existing_count = collection.count()

    if existing_count == 0:
        # 첫 실행: FAQ 로드 (5~10초)
        print("FAQ 데이터 최초 로딩 중...")
        # 모든 항공사 FAQ 미리 로드
        for airline in AIRLINE_FILES.keys():
            try:
                faq_json = load_faq(airline)
                insert_faqs(airline, faq_json)
                print(f"{airline} FAQ 로드 완료")
            except Exception as e:
                print(f"{airline} 로드 실패: {e}")
    else:
        # 재실행: 즉시 사용 (0.5초)
        print(f"기존 FAQ 데이터 사용 중 (총 {existing_count}개)")

    print("="*40 + "\n")
    print("안녕하세요 FLYND입니다.")
    print("무엇을 도와드릴까요?\n")
    
    conversation_history = []  # 대화 히스토리 저장
    last_airline = None  # 마지막으로 사용한 항공사
    
    while True:
        user_question = input("질문 >> ").strip()
        
        if user_question.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        
        if user_question.lower() == "reset":
            conversation_history = []
            last_airline = None
            print("대화가 초기화되었습니다.\n")
            continue
        
        # GPT로 항공사 추출 (대화 히스토리 고려)
        print("질문 분석 중...")
        airlines = extract_airlines(user_question, conversation_history, last_airline)
        
        if not airlines:
            print("항공사를 파악할 수 없습니다. 항공사 이름을 포함해주세요.\n")
            continue
        
        airline = airlines[-1]  # 가장 최근 항공사 선택
        
        if airline != last_airline:
            print(f"{airline} 항공사로 전환되었습니다.")
        
        print(f"답변 생성 중...\n")
        
        answer = generate_answer(user_question, airline, conversation_history)
        print(f"답변: {answer}")
        
        # 대화 히스토리에 추가
        conversation_history.append({
            "user": user_question,
            "bot": answer,
            "airline": airline
        })
        
        last_airline = airline
        print("="*40 + "\n")

