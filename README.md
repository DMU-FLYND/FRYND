# FRYND: 당신의 AI 여행 동반자

FRYND는 당신의 여행 계획을 간소화하기 위해 설계된 지능형 여행 계획 어시스턴트입니다. 거대 언어 모델(LLM)의 강력한 성능과 실시간 여행 데이터를 활용하여, FRYND는 대화형 인터페이스를 통해 항공편 정보를 찾고, 목적지를 탐색하며, 손쉽게 여행을 계획할 수 있도록 돕습니다.

## ✨ 주요 기능

- **대화형 항공편 검색:** "다음 주 인천에서 나리타 가는 비행기 찾아줘"와 같이 자연어로 항공편 정보를 요청할 수 있습니다.
- **실시간 데이터:** 아마데우스(Amadeus) API와 연동하여 최신 항공편 운항 여부 및 가격 정보를 제공합니다.
- **AI 기반 추천:** LangChain을 통해 Google의 Gemini 모델을 활용하여 사용자 쿼리를 이해하고 지능적인 여행 제안을 제공합니다.
- **기내식 문서 RAG Q&A:** 항공사 기내식 PDF를 ChromaDB + Gemini 임베딩으로 인덱싱하여 문서 기반 답변을 제공합니다.
- **사용자 친화적 인터페이스:** Streamlit으로 구축되어 깔끔하고 상호작용이 가능한 채팅 기반 웹 UI를 제공합니다.

## 🛠️ 기술 스택

- **프론트엔드:** Streamlit
- **AI/LLM:** LangChain, Google Gemini
- **여행 데이터:** Amadeus for Developers API
- **핵심 언어:** Python

## 📂 프로젝트 구조

```
/
├── modules/                # 핵심 애플리케이션 로직
│   ├── agent.py            # AI 에이전트 및 도구 정의
│   ├── amadeus.py          # 아마데우스 API 연동 처리
│   ├── chat_ui.py          # Streamlit 사용자 인터페이스 구현
│   ├── meal_chat_ui.py     # 기내식 RAG 전용 채팅 UI
│   ├── meal_rag.py         # 기내식 문서 RAG 파이프라인
│   ├── constants.py        # 설정 및 상수 값 저장
│   └── __init__.py
├── .gitignore              # Git 무시 파일
├── main.py                 # 애플리케이션 실행을 위한 메인 진입점
├── pyproject.toml          # 프로젝트 메타데이터 및 의존성
├── README.md               # 현재 이 파일
└── uv.lock                 # uv 잠금 파일

`data/about_airline_meal.pdf`는 기내식 Q&A 모드를 위한 원본 문서입니다. 최초 실행 시 `data/chroma_meal_db/`에 임베딩 DB가 생성됩니다.
```

## 🚀 시작하기

### 사전 요구사항

- Python 3.12 이상
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip
- Amadeus for Developers API 키 ([여기서 발급](https://developers.amadeus.com/))
- Google AI API 키 ([여기서 발급](https://makersuite.google.com/))

### 설치 방법

1.  **리포지토리 복제:**
    ```bash
    git clone https://github.com/your-username/FRYND.git
    cd FRYND
    ```

2.  **가상 환경 생성 및 의존성 설치:**
    
    uv 사용 시:
    ```bash
    uv venv
    uv sync
    ```
    
    venv 및 pip 사용 시:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt 
    # 참고: pyproject.toml에서 requirements.txt를 생성해야 할 수 있습니다.
    ```

3.  **환경 변수 설정:**

    루트 디렉토리에 `.env` 파일을 생성하고 API 키를 추가합니다:
    ```.env
    AMADEUS_CLIENT_ID="YOUR_AMADEUS_CLIENT_ID"
    AMADEUS_CLIENT_SECRET="YOUR_AMADEUS_CLIENT_SECRET"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

### 애플리케이션 실행

1.  **가상 환경 활성화:**
    ```bash
    source .venv/bin/activate
    ```
2.  **Streamlit 앱 실행:**
    ```bash
    streamlit run main.py
    ```

3.  웹 브라우저를 열고 `http://localhost:8501`로 이동합니다.

### 기내식 RAG 모드 사용 팁

- 상단 라디오 버튼에서 `기내식 Q&A`를 선택하면 문서 기반 질의응답 모드가 활성화됩니다.
- `data/about_airline_meal.pdf`를 교체하면 다음 실행 시 자동으로 새로 임베딩해 반영합니다.
- 최초 임베딩 생성 후에는 `data/chroma_meal_db/`가 캐시 역할을 하므로 빌드 시간이 단축됩니다.

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
