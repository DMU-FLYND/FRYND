"""FAQ pipeline that now uses LangChain + Gemini instead of OpenAI."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import chromadb
from chromadb.errors import NotFoundError
from dotenv import find_dotenv, load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .meal_rag import GeminiEmbeddingFunction

load_dotenv(find_dotenv(), override=True)

# 항공사별 JSON 파일 매핑
AIRLINE_FILES: Dict[str, str] = {
    "진에어": "jinair.json",
    "에어부산": "airbusan.json",
    "티웨이": "tway.json",
    "제주": "jeju.json",
    "에어프레미아": "airpremia.json",
}

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = BASE_DIR / "data" / "chroma_faq_db"
FAQ_DATA_DIR = BASE_DIR / "data" / "faq_data"
COLLECTION_NAME = "airline_faq"
EMBEDDING_MODEL = "gemini-embedding-001"


def load_faq(airline_name: str) -> Dict:
    airline_name = airline_name.strip()

    if airline_name not in AIRLINE_FILES:
        raise ValueError(f"지원하지 않는 항공사입니다: {airline_name}")

    file_path = FAQ_DATA_DIR / AIRLINE_FILES[airline_name]
    if not file_path.exists():
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        faq_data = json.load(f)

    return faq_data


@lru_cache(maxsize=1)
def _get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Google Gemini API 자격증명이 없습니다. 환경 변수 GOOGLE_API_KEY를 설정해 주세요.")
    return api_key


@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=1024,
        api_key=_get_google_api_key(),
    )


@lru_cache(maxsize=1)
def _get_embedding_function() -> GeminiEmbeddingFunction:
    return GeminiEmbeddingFunction(api_key=_get_google_api_key())


@lru_cache(maxsize=1)
def _get_collection():
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    metadata = {"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL}

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
        )
    except NotFoundError:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata=metadata,
            embedding_function=_get_embedding_function(),
        )
    except ValueError as exc:
        message = str(exc).lower()
        conflict = "embedding function already exists" in message or "embedding function conflict" in message
        if conflict:
            collection = _recreate_collection_with_existing_docs(client, metadata)
        else:
            raise

    # 기존 OpenAI 임베딩이 남아 있을 수 있어, 모델 정보가 다르면 새로 임베딩한다.
    if (collection.metadata or {}).get("embedding_model") != EMBEDDING_MODEL:
        collection = _recreate_collection_with_existing_docs(client, metadata)

    # 컬렉션이 비어 있으면 JSON 데이터로 채운다.
    if collection.count() == 0:
        docs, metas, ids = _load_all_faq_documents()
        if docs:
            _safe_add_documents(collection, docs, metas, ids)

    return collection


def _recreate_collection_with_existing_docs(client: chromadb.ClientAPI, metadata: Dict) -> chromadb.Collection:
    """Handle embedding-function conflicts by rebuilding the collection with Gemini embeddings."""

    docs, metas, ids = _load_all_faq_documents()
    existing_docs: List[str] = []
    existing_metas: List[Dict] = []
    existing_ids: List[str] = []

    # JSON 로드에 실패하면 기존 컬렉션 내용을 최대한 보존한다.
    if not docs:
        try:
            stale = client.get_collection(name=COLLECTION_NAME)
            total = stale.count()
            if total:
                fetched = stale.get(include=["documents", "metadatas", "ids"], limit=total)
                existing_docs = fetched.get("documents") or []
                existing_metas = fetched.get("metadatas") or []
                existing_ids = fetched.get("ids") or []
        except Exception:
            pass

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=metadata,
        embedding_function=_get_embedding_function(),
    )

    if docs:
        _safe_add_documents(collection, docs, metas, ids)
    elif existing_docs:
        _safe_add_documents(
            collection,
            existing_docs,
            existing_metas or [{} for _ in existing_docs],
            existing_ids or [f"faq-{i}" for i in range(len(existing_docs))],
        )

    return collection


def insert_faqs(airline: str, faq_data: Dict) -> None:
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    for idx, item in enumerate(faq_data["faqs"]):
        content = item["question"] + " " + item["answer"]
        documents.append(content)
        metadatas.append({"airline": airline})
        ids.append(f"{airline}_{idx}")

    _safe_add_documents(_get_collection(), documents, metadatas, ids)


def _load_all_faq_documents() -> tuple[List[str], List[Dict], List[str]]:
    """Load all FAQ documents from data/faq_data for (re)embedding."""

    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []
    ids: List[str] = []

    for airline, filename in AIRLINE_FILES.items():
        file_path = FAQ_DATA_DIR / filename
        if not file_path.exists():
            continue

        try:
            faq_data = load_faq(airline)
        except Exception:
            # Skip only the failing airline to continue populating others.
            continue

        for idx, item in enumerate(faq_data.get("faqs", [])):
            content = (item.get("question", "") or "") + " " + (item.get("answer", "") or "")
            documents.append(content)
            metadatas.append({"airline": airline})
            ids.append(f"{airline}_{idx}")

    return documents, metadatas, ids


def _safe_add_documents(collection: chromadb.Collection, docs: List[str], metas: List[Dict], ids: List[str]) -> None:
    """Add documents to Chroma with user-friendly handling of Gemini quota errors."""

    try:
        collection.add(documents=docs, metadatas=metas, ids=ids)
    except ResourceExhausted as exc:
        raise RuntimeError(
            "Gemini 임베딩 호출 중 쿼터가 초과되었습니다. Google API 사용량/빌링을 확인하거나, "
            "잠시 후 다시 시도해 주세요."
        ) from exc


def is_airline_mentioned(question: str) -> bool:
    airline_keywords = ["진에어", "에어부산", "티웨이", "제주", "에어프레미아"]
    lowered = question.lower()
    return any(keyword in lowered for keyword in airline_keywords)


def _history_to_text(history: Sequence[Dict[str, str]]) -> str:
    return "\n".join([f"사용자: {h['user']}\n봇: {h['bot']}" for h in history])


@lru_cache(maxsize=1)
def _get_airline_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "사용자의 질문에서 항공사를 파악하세요. 지원 항공사: 진에어, 에어부산, 티웨이, 제주항공, 에어프레미아.",
            ),
            (
                "human",
                "최근 대화 내역:\n{history}\n\n"
                "현재 질문: {question}\n이전 항공사: {last_airline}\n\n"
                "규칙:\n- 질문에 항공사 이름이 있으면 그 항공사만 답변\n"
                "- 없으면 이전 항공사 유지\n- 여러 항공사 비교 시 모두 답변 (쉼표 구분)\n\n"
                "항공사명만 쉼표로 구분해 답변:",
            ),
        ]
    )
    return prompt | _get_llm() | StrOutputParser()


def extract_airlines(
    question: str, conversation_history: Sequence[Dict[str, str]], last_airline: str | None = None
):
    if is_airline_mentioned(question):
        conversation_history = []

    history_text = _history_to_text(conversation_history[-3:])
    airline_text = _get_airline_chain().invoke(
        {
            "history": history_text or "없음",
            "question": question,
            "last_airline": last_airline or "없음",
        }
    )

    airlines = [a.strip() for a in airline_text.split(",") if a.strip()]
    valid_airlines = [a for a in airlines if a in AIRLINE_FILES]
    return valid_airlines if valid_airlines else ([last_airline] if last_airline else None)


@lru_cache(maxsize=1)
def _get_keyword_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '사용자 질문에서 핵심 키워드만 추출하세요. 예: "진에어 수하물 무게 제한이 어떻게 되나요?" -> "수하물, 무게, 제한"',
            ),
            ("human", "질문: {question}\n\n핵심 키워드만 쉼표로 구분하여 답변:"),
        ]
    )
    return prompt | _get_llm() | StrOutputParser()


def extract_keywords(question: str) -> str:
    return _get_keyword_chain().invoke({"question": question}).strip()


@lru_cache(maxsize=1)
def _get_answer_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 {airline} 항공사 고객센터 상담원입니다. 제공된 FAQ 정보를 정확히 참고하여 답변하세요."),
            (
                "human",
                "최근 대화 요약:\n{history}\n\n"
                "{airline} 항공사의 관련 FAQ:\n{faq_context}\n\n"
                "추출된 핵심 키워드: {keywords}\n\n"
                "사용자 질문:\n{question}\n\n"
                "답변 가이드:\n"
                "1. FAQ에 있는 구체적인 정보(금액, 기간, 절차 등)는 그대로 안내\n"
                "2. 여러 FAQ에 분산된 정보는 종합하여 완전하게 답변\n"
                '3. FAQ에 "홈페이지 참고"만 있으면 일반 정보와 함께 안내\n'
                "4. 고객센터 상담원처럼 정중하고 친절하게 답변\n"
                "5. 이전 대화 맥락을 고려해 자연스럽게 답변\n"
                "6. 300자 이내로 간결하게 답변\n"
                '7. FAQ 정보에 링크가 있으면 답변 하단에 함께 표시\n'
                '8. "**" 등 강조 표시는 사용하지 않음\n'
                '9. FAQ에 정보가 없으면 "죄송합니다, 관련 정보를 찾을 수 없습니다."라고 알리고 각 항공사 고객센터 연락처 안내\n\n'
                "최종 답변:",
            ),
        ]
    )
    return prompt | _get_llm() | StrOutputParser()


def generate_answer(question: str, airline: str, conversation_history: Sequence[Dict[str, str]]) -> str:
    keywords = extract_keywords(question)
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
    search_query = question + " " + " ".join(keyword_list)

    num_results = max(5, len(keyword_list) * 3)
    num_results = min(num_results, 15)

    results = _get_collection().query(
        query_texts=[search_query],
        n_results=num_results,
        where={"airline": airline},
    )

    retrieved_docs = (results.get("documents") or [[]])[0] or []
    retrieved_distances = (results.get("distances") or [[]])[0] or []

    filtered_docs: List[str] = []
    for idx, doc in enumerate(retrieved_docs):
        if idx < len(retrieved_distances):
            distance = retrieved_distances[idx]
            if distance < 0.7:
                filtered_docs.append(doc)
        else:
            filtered_docs.append(doc)

    if not filtered_docs:
        return f"죄송합니다. {airline} 항공사의 '{keywords}' 관련 정보를 찾을 수 없습니다."

    faq_context = "\n\n".join([f"[FAQ {i+1}]\n{doc}" for i, doc in enumerate(filtered_docs)])
    history_text = _history_to_text(conversation_history[-3:])

    return _get_answer_chain().invoke(
        {
            "airline": airline,
            "faq_context": faq_context,
            "keywords": keywords,
            "question": question,
            "history": history_text or "없음",
        }
    )


def get_faq_response(question: str) -> str:
    """Streamlit UI에서 호출할 함수"""
    import streamlit as st

    if "faq_conversation_history" not in st.session_state:
        st.session_state.faq_conversation_history = []
    if "faq_last_airline" not in st.session_state:
        st.session_state.faq_last_airline = None

    conversation_history = st.session_state.faq_conversation_history
    last_airline = st.session_state.faq_last_airline

    try:
        airlines = extract_airlines(question, conversation_history, last_airline)
        if not airlines:
            return "항공사를 파악할 수 없습니다. 항공사 이름(진에어, 에어부산, 티웨이, 제주, 에어프레미아)을 포함해주세요."

        airline = airlines[-1]
        answer = generate_answer(question, airline, conversation_history)
    except RuntimeError as exc:
        return f"FAQ 벡터 검색 중 오류가 발생했습니다: {exc}"

    st.session_state.faq_conversation_history.append(
        {
            "user": question,
            "bot": answer,
            "airline": airline,
        }
    )
    st.session_state.faq_last_airline = airline

    return answer


if __name__ == "__main__":
    collection = _get_collection()
    existing_count = collection.count()

    if existing_count == 0:
        print("FAQ 데이터 최초 로딩 중...")
        for airline in AIRLINE_FILES.keys():
            try:
                faq_json = load_faq(airline)
                insert_faqs(airline, faq_json)
                print(f"{airline} FAQ 로드 완료")
            except Exception as e:  # noqa: BLE001
                print(f"{airline} 로드 실패: {e}")
    else:
        print(f"기존 FAQ 데이터 사용 중 (총 {existing_count}개)")

    print("=" * 40 + "\n")
    print("안녕하세요 FLYND입니다.")
    print("무엇을 도와드릴까요?\n")

    conversation_history: List[Dict[str, str]] = []
    last_airline: str | None = None

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

        print("질문 분석 중...")
        try:
            airlines = extract_airlines(user_question, conversation_history, last_airline)
        except RuntimeError as exc:
            print(f"FAQ 벡터 검색 중 오류: {exc}")
            continue

        if not airlines:
            print("항공사를 파악할 수 없습니다. 항공사 이름을 포함해주세요.\n")
            continue

        airline = airlines[-1]

        if airline != last_airline:
            print(f"{airline} 항공사로 전환되었습니다.")

        print("답변 생성 중...\n")

        try:
            answer = generate_answer(user_question, airline, conversation_history)
        except RuntimeError as exc:
            print(f"FAQ 벡터 검색 중 오류: {exc}")
            continue
        print(f"답변: {answer}")

        conversation_history.append(
            {
                "user": user_question,
                "bot": answer,
                "airline": airline,
            }
        )

        last_airline = airline
        print("=" * 40 + "\n")
