import streamlit as st
from dotenv import load_dotenv
from src.config import *
from src.data_loader import *
from src.models import *
from src.retrievers import *
from src.chatbot import *
from src.ui import *
from src.prompts import get_chat_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from src.ui import (
    initialize_streamlit_ui,
    display_messages,
    handle_streamlit_input,
)


def load_environment():
    load_dotenv()


def main():
    initialize_streamlit_ui()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        ]

    display_messages()

    # JSON 파일 경로 설정
    file_paths = {
        "mct": "/data/mct.json",
        "month": "/data/month.json",
        "wkday": "/data/wkday.json",
        "mop_sentiment": "/data/merge_mop_sentiment.json",
        "menu": "/data/mct_menus.json",
        "visit_jeju": "/data/visit_jeju.json",
        "kakaomap_reviews": "/data/kakaomap_reviews.json",
    }

    # FAISS 인덱스 경로 설정
    index_paths = {
        "mct": "/data/faiss_index/mct_index_pq.faiss",
        "month": "/data/faiss_index/month_index_pq.faiss",
        "wkday": "/data/faiss_index/wkday_index_pq.faiss",
        "menu": "/data/faiss_index/menu.faiss",
        "visit": "/data/faiss_index/visit_jeju.faiss",
        "kakaomap_reviews": "/data/faiss_index/kakaomap_reviews.faiss",
    }

    # 데이터 및 인덱스 로드
    data = load_json_files(file_paths)
    faiss_indexes = load_faiss_indexes(index_paths)

    # Document 및 Embedding 초기화
    docs = create_documents(data)
    embedding, tokenizer, model = initialize_embeddings()

    # Retriever 및 Ensemble Retriever 초기화
    retrievers = initialize_faiss_retrievers(docs, embedding)
    bm25_retrievers = initialize_bm25_retrievers(docs)
    ensemble_retrievers = initialize_ensemble_retrievers(retrievers, bm25_retrievers)

    # LLM Chain 및 Memory 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = get_chat_prompt()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,  # 더 낮은 temperature로 설정해 할루시네이션 줄임
        top_p=0.85,  # top_p를 조정해 더 예측 가능한 답변 생성
        frequency_penalty=0.1,  # 같은 단어의 반복을 줄이기 위해 패널티 추가
    )

    chain = LLMChain(prompt=prompt_template, llm=llm, output_parser=StrOutputParser())

    # UI 메시지 출력 및 입력 처리
    display_messages()
    handle_streamlit_input(chain, memory)


if __name__ == "__main__":
    main()
