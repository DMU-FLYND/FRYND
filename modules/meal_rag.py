"""RAG helpers for answering in-flight meal questions based on the onboard PDF."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.errors import NotFoundError
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent.parent
MEAL_DOC_PATH = BASE_DIR / "data" / "about_airline_meal.pdf"
CHROMA_DIR = BASE_DIR / "data" / "chroma_meal_db"
COLLECTION_NAME = "airline_meal_docs"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function that uses Gemini embeddings for ChromaDB."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        genai.configure(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        if not input:
            return []

        embeddings: Embeddings = []
        for text in input:
            content = text or ""
            response = genai.embed_content(
                model="gemini-embedding-001",
                content=content,
                task_type="retrieval_document",
                title="airline_meal",
            )
            embeddings.append(response["embedding"])
        return embeddings


def answer_meal_question(query: str, history: Sequence[BaseMessage], top_k: int = 3) -> str:
    """Run the full RAG pipeline and return an answer for the user's question."""

    llm = _build_llm()
    refined_query = _get_query_refiner(llm).invoke({"messages": list(history), "query": query})
    context = retrieve_meal_passages(refined_query, top_k=top_k)

    if not context.strip():
        return "관련 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 입력해 주세요."

    return _get_answer_chain(llm).invoke({"context": context, "question": refined_query})


def retrieve_meal_passages(query: str, top_k: int = 3) -> str:
    """Retrieve the most relevant passages from the meal document."""

    collection = get_meal_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    documents: List[str] = results.get("documents", [[]])[0] or []
    cleaned = [doc.strip() for doc in documents if doc]
    return "\n\n".join(cleaned)


@lru_cache(maxsize=1)
def get_meal_collection():
    """Load or build the ChromaDB collection for the in-flight meal document."""

    api_key = _get_google_api_key()
    embedding_fn = GeminiEmbeddingFunction(api_key)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    except NotFoundError:
        pass
    except ValueError:
        # 기존 컬렉션이 깨졌거나 다른 임베딩 함수로 생성된 경우 재생성한다.
        client.delete_collection(name=COLLECTION_NAME)

    documents = _load_meal_documents()
    if not documents:
        raise RuntimeError(f"문서를 불러오지 못했습니다. 파일 위치를 확인하세요: {MEAL_DOC_PATH}")

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    ids = [f"meal-{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)
    return collection


def _load_meal_documents() -> List[str]:
    if not MEAL_DOC_PATH.exists():
        raise RuntimeError(
            f"기내식 문서가 없습니다. 파일을 다음 경로에 두고 다시 시도하세요: {MEAL_DOC_PATH}"
        )

    loader = PyPDFLoader(str(MEAL_DOC_PATH))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(pages)
    return [doc.page_content for doc in splits]


def _get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Google API 키가 설정되지 않았습니다. 환경 변수 GOOGLE_API_KEY를 확인하세요.")
    return api_key


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=1024,
    )


def _get_query_refiner(llm: ChatGoogleGenerativeAI):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 질문을 재작성하는 도우미이다. 대화 내용을 참고해 사용자의 마지막 질문의 의미를 명확한 한 문장으로"
                " 재작성하라. 대명사(이, 저, 그 등)는 구체적 명사로 바꿔라.",
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
                "너는 항공사별 기내식 관련 질문에 답하는 전문가다. 제공된 문서 내용만을 근거로 한국어로 답하라. "
                "문서에 없는 내용은 임의로 추측하지 말고, 충분한 정보가 없다고 말해라.",
            ),
            (
                "human",
                "아래는 참고 문서이다. 문서 내용만 근거로 사용자의 질문에 답하라.\n\n"
                "[문서]\n{context}\n\n[질문]\n{question}\n\n답변:",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()


__all__ = [
    "answer_meal_question",
    "retrieve_meal_passages",
    "get_meal_collection",
    "GeminiEmbeddingFunction",
]
