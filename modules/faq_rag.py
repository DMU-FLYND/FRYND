"""RAG helpers for answering airline FAQ questions from JSON data."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import chromadb
from chromadb.errors import NotFoundError
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from .meal_rag import GeminiEmbeddingFunction

BASE_DIR = Path(__file__).resolve().parent.parent
FAQ_DATA_DIR = BASE_DIR / "data" / "faq_data"
CHROMA_DIR = BASE_DIR / "data" / "chroma_faq_db"
COLLECTION_NAME = "airline_faq_docs"


def answer_faq_question(
    query: str,
    history: Sequence[BaseMessage],
    top_k: int = 4,
    airline: str | None = None,
) -> str:
    """Run the FAQ RAG pipeline and return an answer."""

    llm = _build_llm()
    airline_hint = airline or "전체"
    refined_query = _get_query_refiner(llm).invoke(
        {"messages": list(history), "query": query, "airline": airline_hint}
    )
    context = retrieve_faq_passages(refined_query, top_k=top_k, airline=airline)

    if not context.strip():
        target = f"{airline_hint} FAQ"
        return f"{target}에서 관련 답변을 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."

    return _get_answer_chain(llm).invoke(
        {"context": context, "question": refined_query, "airline": airline_hint}
    )


def retrieve_faq_passages(query: str, top_k: int = 4, airline: str | None = None) -> str:
    """Retrieve the most relevant FAQ passages (optionally filtered by airline)."""

    collection = get_faq_collection()
    where = {"airline": airline} if airline else None
    results = collection.query(query_texts=[query], n_results=top_k, where=where)
    documents: List[str] = results.get("documents", [[]])[0] or []
    cleaned = [doc.strip() for doc in documents if doc]
    return "\n\n".join(cleaned)


@lru_cache(maxsize=1)
def get_faq_collection():
    """Load or build the ChromaDB collection for FAQ documents.

    Falls back to an in-memory Chroma client when the persistent path is not writable.
    """

    api_key = _get_google_api_key()
    embedding_fn = GeminiEmbeddingFunction(api_key)
    client, persistent = _get_chroma_client()

    documents, metadatas = _load_faq_documents()
    ids = [meta["doc_id"] for meta in metadatas]

    def _build_new_collection():
        collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
        _add_documents_in_batches(collection, documents, metadatas, ids)
        return collection

    # Ephemeral client: just build once per process via lru_cache and return.
    if not persistent:
        return _build_new_collection()

    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    except NotFoundError:
        return _build_new_collection()
    except ValueError:
        client.delete_collection(name=COLLECTION_NAME)
        return _build_new_collection()

    current_count = collection.count()
    if current_count == len(documents):
        return collection
    if current_count == 0:
        _add_documents_in_batches(collection, documents, metadatas, ids)
        return collection

    client.delete_collection(name=COLLECTION_NAME)
    return _build_new_collection()


def get_supported_airlines() -> List[str]:
    """Return the list of airlines found in the FAQ JSON files."""

    airlines = {entry["airline"] for entry in _load_raw_faq_entries()}
    return sorted(airlines)


def _add_documents_in_batches(collection, documents: List[str], metadatas: List[Dict], ids: List[str], batch_size: int = 100) -> None:
    """Add documents to ChromaDB collection in batches to avoid API rate limits."""
    
    total = len(documents)
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                # API 호출 제한을 피하기 위해 배치 사이에 짧은 대기
                if i + batch_size < total:
                    time.sleep(1)
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(f"FAQ 문서 추가 실패 (배치 {i//batch_size + 1}): {e}")
                # 재시도 전에 지수 백오프
                time.sleep(2 ** retry_count)


def _load_faq_documents() -> Tuple[List[str], List[Dict[str, str]]]:
    entries = _load_raw_faq_entries()
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for idx, entry in enumerate(entries):
        tag_text = ", ".join(entry["tags"]) if entry["tags"] else "태그 없음"
        document = (
            f"[항공사] {entry['airline']}\n"
            f"[질문] {entry['question']}\n"
            f"[답변] {entry['answer']}\n"
            f"[키워드] {tag_text}"
        )
        documents.append(document)
        metadatas.append(
            {
                "airline": entry["airline"],
                "question": entry["question"],
                "source": entry["source"],
                "tags": tag_text,
                "doc_id": f"{entry['source']}-{idx}",
            }
        )

    if not documents:
        raise RuntimeError(f"FAQ 데이터를 찾을 수 없습니다. 경로를 확인하세요: {FAQ_DATA_DIR}")

    return documents, metadatas


def _load_raw_faq_entries() -> List[Dict[str, object]]:
    if not FAQ_DATA_DIR.exists():
        raise RuntimeError(
            f"FAQ 데이터 디렉토리가 없습니다. 파일 위치를 확인하세요: {FAQ_DATA_DIR}"
        )

    entries: List[Dict[str, object]] = []
    for path in sorted(FAQ_DATA_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        airline = str(data.get("airline") or path.stem).strip()
        for faq in data.get("faqs", []):
            question = str(faq.get("question", "")).strip()
            answer = str(faq.get("answer", "")).strip()
            tags = [str(tag).strip() for tag in faq.get("tags", []) if str(tag).strip()]
            entries.append(
                {
                    "airline": airline,
                    "question": question,
                    "answer": answer,
                    "tags": tags,
                    "source": path.stem,
                }
            )

    return entries


def _get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Google API 키가 설정되지 않았습니다. 환경 변수 GOOGLE_API_KEY를 확인하세요.")
    return api_key


def _get_chroma_client():
    """Return a Chroma client with a persistent store if possible; otherwise in-memory."""

    try:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client, True
    except OSError:
        # 파일 시스템 권한 문제 등으로 영속 저장이 불가한 경우 메모리 모드로 대체한다.
        client = chromadb.EphemeralClient()
        return client, False


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=5000,
    )


def _get_query_refiner(llm: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 항공사 FAQ 질문을 재작성하는 도우미이다. 대화 내용을 참고해 사용자의 마지막 질문을 "
                "지정된 항공사 맥락({airline})에 맞는 명확한 한 문장으로 재작성하라. "
                "대명사(이, 저, 그 등)는 구체적 명사로 바꿔라.",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "재작성할 질문: {query}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _get_answer_chain(llm: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 항공사 FAQ에 답하는 전문가다. 제공된 FAQ 문맥만 근거로 한국어로 답하라. "
                "문맥에 없는 내용은 추측하지 말고, 충분한 정보가 없다고 말해라. "
                "여러 항공사의 문맥이 있는 경우 질문과 가장 관련된 항공사명을 함께 언급하라.",
            ),
            (
                "human",
                "아래는 참고 FAQ 문서이다. 문서 내용만 근거로 사용자의 질문에 답하라.\n\n"
                "[문서]\n{context}\n\n[질문]\n{question}\n\n답변:",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


__all__ = [
    "answer_faq_question",
    "retrieve_faq_passages",
    "get_faq_collection",
    "get_supported_airlines",
]
