# FRYND: 당신의 AI 여행 동반자

FRYND는 항공권 검색, 항공사별 FAQ, 기내식 정보를 **하나의 통합 챗봇**으로 제공하는 Streamlit 애플리케이션입니다. LangChain과 Google Gemini (Generative AI)를 통해 자연스러운 한국어 대화를 처리하고, Amadeus API와 자체 RAG 파이프라인을 조합해 실시간·문서 기반 정보를 제공합니다.

## ✨ 주요 기능

- **통합 대화 라우팅:** 한 입력창에서 항공권·FAQ·기내식 질문을 받으면 키워드 기반으로 자동 분기해 가장 적합한 툴을 호출합니다.
- **실시간 항공권 탐색:** Amadeus API로 ICN·GMP·HND·NRT 구간의 편도/왕복 운임을 조회하고 표 형태로 응답합니다.
- **FAQ & 기내식 RAG:** 항공사 FAQ JSON과 `about_airline_meal.pdf`를 ChromaDB + Gemini 임베딩으로 인덱싱해 정답을 생성합니다.
- **AI 음성 합성:** 어시스턴트 답변마다 🔊 버튼을 제공하며, Gemini 2.5 Flash TTS 프리뷰 모델을 우선 사용하고 실패 시 gTTS로 폴백합니다.
- **Streamlit UI:** 반응형 레이아웃, 사이드바 필터, 표/오디오 출력 등 여행 상담에 특화된 UX를 제공합니다.

## 🛠️ 기술 스택

- **언어 / 런타임:** Python 3.12, uv (권장)
- **LLM & 에이전트:** LangChain, Gemini 2.5 Flash (chat + embedding + TTS)
- **데이터 소스:** Amadeus for Developers Flight Offers Search API
- **벡터 스토어:** ChromaDB (FAQ & Meal) + Gemini 임베딩
- **프레임워크:** Streamlit, Pydantic, Pandas, dotenv

## 📂 프로젝트 구조

```
FRYND/
├── main.py                  # Streamlit 진입점 (unified chatbot)
├── modules/
│   ├── unified_chat_ui.py   # 항공권/FAQ/기내식 통합 UI & 라우팅
│   ├── agent.py             # LangChain 도구 기반 항공권 에이전트
│   ├── amadeus.py           # Amadeus API 호출 및 응답 정규화
│   ├── faq_rag.py           # FAQ RAG 파이프라인 및 Chroma 관리
│   ├── meal_rag.py          # 기내식 PDF RAG 파이프라인
│   ├── tts_helper.py        # Gemini/gTTS TTS 헬퍼
│   ├── constants.py         # 공항/항공사 레퍼런스 정보
│   └── ...
├── data/
│   ├── about_airline_meal.pdf   # 기내식 RAG 원본 문서
│   ├── faq_data/*.json          # 항공사 FAQ 데이터셋
│   ├── chroma_faq_db/           # FAQ 인덱스 (자동 생성)
│   └── chroma_meal_db/          # 기내식 인덱스 (자동 생성)
├── notebook/                # 데이터/웹스크래핑 노트북
├── pyproject.toml           # uv/PEP 621 메타데이터
├── requirements.txt         # pip용 잠금 스냅샷
├── uv.lock                  # uv 잠금 파일
└── README.md
```

## ⚙️ 동작 구성요소

- `modules/unified_chat_ui.py`: 세션 상태와 사이드바 필터를 관리하고, 입력을 **항공권 / 기내식 / FAQ**로 분류합니다.
- `modules/agent.py`: Gemini LLM + LangChain 도구 호출 에이전트. `flight_offer_lookup` 도구를 통해 Amadeus 데이터를 받아 표 형태로 변환합니다.
- `modules/faq_rag.py` & `modules/meal_rag.py`: Gemini 임베딩으로 ChromaDB 인덱스를 구축/갱신하고, 문맥 기반 답변을 생성합니다.
- `modules/tts_helper.py`: Google `google-genai` 클라이언트와 gTTS를 결합한 음성 합성 출력.

## 🚀 시작하기

### 1. 사전 요구사항

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 0.4+ (권장) 또는 표준 `venv + pip`
- Amadeus for Developers API (Client ID/Secret, Sandbox 권한)
- Google AI Studio API 키 (Gemini Chat + Embedding + TTS 사용)

### 2. 설치

```bash
git clone https://github.com/DMU-FLYND/FRYND.git
cd FRYND

# uv 권장 경로
uv venv
uv sync

# --- 또는 pip ---
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 환경 변수

루트에 `.env` 파일을 만들고 다음 값을 채웁니다.

| 변수 | 필수 | 설명 |
| --- | --- | --- |
| `GOOGLE_API_KEY` | ✅ | Gemini Chat/RAG/TTS에 사용되는 API 키 |
| `AMADEUS_CLIENT_ID` | ✅ | Amadeus Flight Offers Search 클라이언트 ID |
| `AMADEUS_CLIENT_SECRET` | ✅ | Amadeus 클라이언트 시크릿 |
| `AMADEUS_HOSTNAME` | ⛔️ | 기본 `test`. 실서비스 전환 시 `production`으로 변경 |
| `GEMINI_TTS_MODEL` | ⛔️ | 기본 `gemini-2.5-flash-preview-tts` |
| `GEMINI_TTS_VOICE` | ⛔️ | 기본 `Kore`. 다른 사전 구성 음성 이름으로 교체 가능 |

### 4. 실행

```bash
source .venv/bin/activate        # uv 사용 시 uv run streamlit run main.py 로 대체 가능
streamlit run main.py
# 브라우저: http://localhost:8501
```

첫 실행 시 FAQ/기내식 RAG 인덱스가 `data/chroma_*` 폴더에 생성되므로 몇 초 정도 추가 시간이 걸릴 수 있습니다.

## 📚 데이터 & RAG 인덱스 관리

- **기내식 문서 교체:** `data/about_airline_meal.pdf`를 새 PDF로 교체 후 `data/chroma_meal_db/` 폴더를 삭제하면 다음 실행 때 자동으로 다시 임베딩합니다.
- **FAQ 데이터 추가:** `data/faq_data/`에 `airline_name.json` 형식으로 항공사와 FAQ 배열을 추가하면, 앱이 재시작될 때 자동으로 인덱스를 재구성합니다.
- **벡터 DB 리셋:** 인덱스가 손상된 경우 `data/chroma_faq_db/` 또는 `data/chroma_meal_db/`를 삭제하고 앱을 재시작해 새롭게 빌드합니다.

## 🧪 개발 팁

- `notebook/chatbot_meal.ipynb`, `notebook/web_scraping_meal.ipynb`를 활용해 데이터 수집·전처리를 반복할 수 있습니다.
- Streamlit 캐시(`@st.cache_resource`)가 적용된 LangChain Executor가 있으므로, 파라미터 변경 후에는 **새로고침** 또는 `st.cache_resource.clear()` 호출로 캐시를 초기화하세요.
- Google GenAI, gTTS 등 일부 모듈은 런타임 import가 실패하면 명확한 예외 메시지를 표시하므로, 로그를 확인해 누락된 패키지를 설치합니다 (`uv add google-genai` 등).

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
