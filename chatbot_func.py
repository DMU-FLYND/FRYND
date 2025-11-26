import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 참고 : https://github.com/google-gemini/cookbook/blob/main/examples/chromadb/Vectordb_with_chroma.ipynb

load_dotenv('key.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

loader = PyPDFLoader('D:\chatbot_project\\about_airline_meal.pdf')
data_nyc = loader.load()

text_splitter =  RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap = 100)
all_splits = text_splitter.split_documents(data_nyc)

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    title = "Custom query"
    response = genai.embed_content(
        model='gemini-embedding-001',
        content=input,
		task_type="retrieval_document",
		title=title
    )
    return response['embedding']

folder_path = 'D:\chatbot_project\chroma_db'

def create_chroma_db(documents, name):
  chroma_client = chromadb.PersistentClient(path=folder_path)
  db = chroma_client.create_collection(
      name=name,
      embedding_function=GeminiEmbeddingFunction()
  )

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db

documents = [i.page_content for i in all_splits]

db_pos = 'chroma_db'
try:
	db = create_chroma_db(documents, db_pos) # 생성을 하지 않은 경우 DB 생성 
except:
    pass

chroma_client = chromadb.PersistentClient(path=folder_path)

# 이미 생성한 경우 2번 수행할 필요가 없기에 기존 DB 불러오기
db = chroma_client.get_collection(
    name=db_pos, 
    embedding_function=GeminiEmbeddingFunction()
)

sample_data = db.get(include=['documents', 'embeddings'])

df = pd.DataFrame({
    "IDs": sample_data['ids'][:3],
    "Documents": sample_data['documents'][:3],
    "Embeddings": [str(emb)[:50] + "..." for emb in sample_data['embeddings'][:3]]  # Truncate embeddings
})

def get_relevant_passage(query, db, n):
  passage = db.query(query_texts=[query], n_results=n)['documents'][0]
  return ''.join(passage)

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""
    너는 지금부터 항공사별 기내식 관련 질문에 답하는 전문가야. 
    다음 문서를 기반으로 정확하게 응답해줘. 
    항공사는 총 5개고 명확히 명시되어있지 않은 마지막 항공사는 tway 항공사야.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'""").format(query=query, relevant_passage=escaped)

  return prompt

model = genai.GenerativeModel("gemini-2.5-flash")

query = "원하는 prompt 입력하기"
passage = get_relevant_passage(query, db, 3)
prompt = make_prompt(query, passage)

response = model.generate_content(prompt)
# print(response.text) 답변 text만 추출 가능