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

    file_path = os.path.join(os.path.dirname(__file__), "data", AIRLINE_FILES[airline_name])

    with open(file_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    return faq_data


# ChromaDB 벡터 DB 초기화
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="airline_faq",
    metadata={"hnsw:space": "cosine"}
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


# GPT로 질문에서 항공사 추출 (여러 항공사 가능)
def extract_airlines(question, conversation_history, last_airline=None):
    history_text = "\n".join([f"사용자: {h['user']}\n봇: {h['bot']}" for h in conversation_history[-3:]])
    
    prompt = f"""
사용자의 질문에서 항공사를 파악하세요.
지원 항공사: 진에어, 에어부산, 티웨이, 제주, 에어프레미아

최근 대화 내역:
{history_text}

현재 질문: {question}
이전 항공사: {last_airline if last_airline else "없음"}

중요:
- 여러 항공사를 비교하는 질문이면 모든 항공사를 쉼표로 구분하여 답변하세요. 예: "티웨이,제주"
- 항공사가 하나만 언급되면 그 항공사만 답변하세요.
- 파악할 수 없으면 "알 수 없음"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    airline_text = response.choices[0].message.content.strip()
    airlines = [a.strip() for a in airline_text.split(",")]
    valid_airlines = [a for a in airlines if a in AIRLINE_FILES]
    
    return valid_airlines if valid_airlines else ([last_airline] if last_airline else None)


# 검색 + gpt-4o-mini 답변 생성 (대화 히스토리 포함)
def generate_answer(question, airline, conversation_history):
    # 항공사 필터링 검색
    results = collection.query(
        query_texts=[question],
        n_results=3,
        where={"airline": airline}
    )

    retrieved_docs = results["documents"][0]
    
    # 대화 히스토리를 메시지 형태로 변환
    messages = [
        {"role": "system", "content": f"당신은 {airline} 항공사 고객센터 상담원입니다."}
    ]
    
    # 최근 3턴의 대화 추가
    for hist in conversation_history[-3:]:
        messages.append({"role": "user", "content": hist["user"]})
        messages.append({"role": "assistant", "content": hist["bot"]})
    
    # 현재 질문과 FAQ 정보
    current_prompt = f"""
아래는 {airline} 항공사의 관련 FAQ 내용입니다:

{retrieved_docs}

사용자 질문:
{question}

위 정보를 참고하여 정확하게 답변해 주세요.
이전 대화 맥락을 고려하여 자연스럽게 답변하세요.
"""
    
    messages.append({"role": "user", "content": current_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


# 메인 실행부
if __name__ == "__main__":
    # 모든 항공사 FAQ 미리 로드
    for airline in AIRLINE_FILES.keys():
        try:
            faq_json = load_faq(airline)
            insert_faqs(airline, faq_json)
            print(f"{airline} FAQ 로드 완료")
        except Exception as e:
            print(f"{airline} 로드 실패: {e}")
    
    print("\n" + "="*40 + "\n")
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
            print("✅ 대화가 초기화되었습니다.\n")
            continue
        
        # GPT로 항공사 추출 (대화 히스토리 고려)
        print("🔍 질문 분석 중...")
        airlines = extract_airlines(user_question, conversation_history, last_airline)
        
        if not airlines:
            print("❌ 항공사를 파악할 수 없습니다. 항공사 이름을 포함해주세요.\n")
            continue
        
        airline = airlines[-1]  # 가장 최근 항공사 선택
        
        if airline != last_airline:
            print(f"✈️  {airline} 항공사로 전환되었습니다.")
        
        print(f"📝 답변 생성 중...\n")
        
        answer = generate_answer(user_question, airline, conversation_history)
        print(f"답변: {answer}\n")
        
        # 대화 히스토리에 추가
        conversation_history.append({
            "user": user_question,
            "bot": answer,
            "airline": airline
        })
        
        last_airline = airline
        print("="*40 + "\n")

