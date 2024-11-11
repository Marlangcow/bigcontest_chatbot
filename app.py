import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st

from transformers import AutoTokenizer, AutoModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from google.cloud import dialogflow_v2 as dialogflow
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
import google.generativeai as genai
from typing import List, Dict
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import faiss
import json
import torch

# 1. 설정 및 상수

# CONFIG 객체를 import문 다음, 다른 코드들 이전에 정의
CONFIG = {
    'model_name': "jhgan/ko-sroberta-multitask",
    'similarity_threshold': 0.7,
    'retriever_weights': [0.6, 0.4],
    'search_params': {
        'k': 4,
        'fetch_k': 10,
        'lambda_mult': 0.6,
        'score_threshold': 0.6
    }
}

# .env 파일 경로 지정
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY_1")

# 상단에 CSS 스타일 정의
STYLES = """
<style>
/* Selectbox 레이블 숨기기 및 여백 조정 */
.stSelectbox label { 
    display: none; 
}
.stSelectbox div[role='combobox'] { 
    margin-top: -20px; 
}

/* Radio button 레이블 숨기기 및 여백 조정 */
.stRadio > label { 
    display: none; 
}
.stRadio > div { 
    margin-top: -20px; 
}
</style>
"""

st.set_page_config(page_title="🍊감귤톡")
st.markdown(STYLES, unsafe_allow_html=True)

# 메인 UI
st.title("🍊감귤톡, 제주도 여행 메이트")
st.write("")
st.info("제주도 여행 메이트 감귤톡이 제주도의 방방곡곡을 알려줄게 🏝️")

# 이미지 로드 설정
image_path = "https://img4.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202105/25/linkagelab/20210525013157546odxh.jpg"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="50%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)

# 대화 초기화 함수 정의
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]
    
# 사이드바 구성
with st.sidebar:
    st.title("🍊감귤톡이 다 찾아줄게🍊")
    st.write("")
    
    st.subheader("원하는 #키워드를 골라봐")
    keywords = st.selectbox(
        "",
        [
            "착한가격업소",
            "럭셔리트래블인제주",
            "우수관광사업체",
            "무장애관광",
            "안전여행스탬프",
            "향토음식",
            "한식",
            "카페",
            "해물뚝배기",
            "몸국",
            "해장국",
            "수제버거",
            "흑돼지",
            "해산물",
            "일식",
        ],
        key="keywords",
    )
    
    st.subheader("어떤 동네가 궁금해?")
    locations = st.selectbox(
        "",
        (
            "구좌",
            "대정",
            "서귀포",
            "안덕",
            "우도",
            "애월",
            "조천",
            "제주시내",
            "추자",
            "한림",
            "한경",
        ),
        key="locations"
    )
    st.write("")

    st.subheader("평점 몇점 이상을 찾고 싶어?")
    score = st.slider(
        "리뷰 평점", 
        min_value=3.0, 
        max_value=5.0, 
        value=4.0, 
        step=0.05,
        key="score"
    )
    
    st.write("")
    st.button("대화 초기화", on_click=clear_chat_history, key="clear_chat_sidebar")
    st.caption("📨 감귤톡에 문의하세요 [Send email](mailto:happily2bus@gmail.com)")


      
# HuggingFaceEmbeddings 객체 초기화
embedding = HuggingFaceEmbeddings(model_name=CONFIG['model_name'])

# Google Generative AI API 설정 부분 이전에 memory 정의
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 그 다음 llm과 chain 정의
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    top_p=0.85,
    frequency_penalty=0.1,
    google_api_key=google_api_key,
    credentials=None
)

PROMPT_TEMPLATE = """
    ### 역할
    당신은 제주도 맛집과 관광지 추천 전문가입니다. 질문을 받을 때 논리적으로 생각한 후 단계별로 답변을 제공합니다.
    복잡한 질문일수록 천천히 생각하고 검색된 데이터를 바탕으로 친근하고 정겨운 답변을 제공합니다.

    ### Chain of Thought 방식 적용:
    1. 사용자의 질문을 단계별로 분석합니다.
    2. 질문의 위치 정보를 파악합니다.
    3. 그 후에 사용자가 제공한 정보나 검색된 데이터를 바탕으로 관련성 있는 맛집과 관광지를 추천합니다.
    4. 단계를 나누어 정보를 체계적으로 제공합니다.

    ### 지시사항
    1. 검색할 내용이 충분하지 않다면 사용자에게 반문하세요. 이는 가장 중요합니다. 단, 두번 이상 반문하지 마세요. 만약 사용자가 위치를 모른다면 제일 평점이 좋은 3개의 식당+카페와 3개의 관광지를 안내해주세요.
    2. 답변을 하는 경우 어떤 문서를 인용했는지 (키:값) 에서 키는 제외하고 값만 답변 뒤에 언급하세요.
      (mct_docs: 신한카드 가맹점 - 요식업, month_docs: 비짓제주 - 월별 조회수, wkday_docs: 비짓제주 - 요일별 조회수, mop_docs: 관광지 평점리뷰, menu_docs: 카카오맵 가게 메뉴, visit_docs: 비짓제주 - 여행지 정보, kakaomap_reviews_docs: 카카오맵 리뷰)
    4. 추천 이유와 거리, 소요 시간, 핵심키워드 3개, 평점과 리뷰들도 보여주세요. 만약 리뷰가 없는 곳이라면 ("아직 작성된 리뷰가 없습니다.") 라고 해주세요.
    5. 4번의 지시사항과 함께 판매 메뉴 2개, 가격을 함께 알려주세요.
    6. 주소를 바탕으로 실제 검색되는 장소를 아래 예시 링크 형식으로 답변하세요.
      - 네이버 지도 확인하기: (https://map.naver.com/p/search/제주도+<place>장소명</place>)
    7. 실제로 존재하는 식당과 관광지명을 추천해주어야 하며, %%흑돼지 맛집, 횟집 1 등 가게명이 명확하지 않은 답변은 하지 말아주세요.
    8. 답변 내용에 따라 폰트사이즈, 불렛, 순서 활용하고 문단을 구분하여 가독성이 좋게 해주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """
    
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template=PROMPT_TEMPLATE
)

# 체인 생성
chain = LLMChain(
    prompt=prompt_template,
    llm=llm,
    output_parser=StrOutputParser(),
)

# RAG

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is {device}.")

# JSON 파일 경로 설정
file_paths = {
    "mct": "/Users/naeun/bigcontest_chatbot/data/mct.json",
    "month": "/Users/naeun/bigcontest_chatbot/data/month.json",
    "wkday": "/Users/naeun/bigcontest_chatbot/data/wkday.json",
    "mop_sentiment": "/Users/naeun/bigcontest_chatbot/data/merge_mop_sentiment.json",
    "menu": "/Users/naeun/bigcontest_chatbot/data/mct_menus.json",
    "visit_jeju": "/Users/naeun/bigcontest_chatbot/data/visit_jeju.json",
    "kakaomap_reviews": "/Users/naeun/bigcontest_chatbot/data/kakaomap_reviews.json",
}

# 2. 초기화 함수들
def initialize_retriever(db):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs=CONFIG['search_params']
    )

# 3. 유틸리티 함수들
# JSON 파일 로드
def load_json_files(file_paths):
    data = {}
    for key, path in file_paths.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data[key] = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {path}")
    return data

# FAISS 인덱스 로드 시 오류 처리 함수 개선
def load_faiss_indexes(index_paths):
    indexes = {}
    for key, path in index_paths.items():
        try:
            if not os.path.exists(path):
                print(f"Warning: Index file not found: {path}")
                continue  # 파일이 없으면 건너뛰기
            indexes[key] = faiss.read_index(path)  # FAISS 인덱스 로드
        except faiss.FaissException as e:
            print(f"FAISS error loading index '{key}': {e}")
        except Exception as e:
            print(f"Unexpected error loading index '{key}': {e}")
    return indexes

# 4. 핵심 기능 함수들

# 통합된 응답 생성 함수를 먼저 정의
def get_chatbot_response(query, memory, chain):
    try:
        # 검색 결과 추출
        search_results = flexible_function_call_search(query)
        if not search_results:
            return "관련된 정보를 찾을 수 없습니다."

        # 검색 결과를 문자열로 변환
        search_results_str = "\n".join(
            [doc.page_content for doc in search_results]
        ).strip()
        if not search_results_str:
            return "검색된 내용이 없어서 답변을 드릴 수 없습니다."

        # 대화 기록 로드
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        # LLMChain에 전달할 입력 데이터 구성
        input_data = {
            "input_text": query,
            "search_results": search_results_str,
            "chat_history": chat_history,
        }

        try:
            output = chain(input_data)
            output_text = output.get("text", str(output))
        except Exception as e:
            print(f"LLM 응답 생성 중 오류 발생: {e}")
            return "응답을 생성하는 과정에서 오류가 발생했습니다. 다시 시도해주세요."

        # 대화 기록에 입력 및 출력 저장
        memory.save_context({"input": query}, {"output": output_text})
        return output_text

    except Exception as e:
        print(f"검색 또는 응답 생성 중 오류 발생: {e}")
        return "오류가 발생했습니다. 다시 시도해주세요."

# 임베딩 캐싱 추가
@st.cache_data(ttl=3600)
def get_embedding(text):
    return embedding.embed_query(text)

def flexible_function_call_search(query):
    try:
        # 입력 쿼리의 임베딩을 가져옵니다.
        input_embedding = get_embedding(query)

        # 리트리버와 설명을 정의
        retrievers_info = {
            "mct": {
                "retriever": mct_retriever,
                "description": "식당 정보 및 연이용 비중 및 금액 비중",
            },
            "month": {
                "retriever": month_retriever,
                "description": "관광지 월별 조회수",
            },
            "wkday": {
                "retriever": wkday_retriever,
                "description": "주별 일별 조회수 및 연령별 성별별 선호도",
            },
            "mop": {
                "retriever": mop_retriever,
                "description": "관광지 전체 감성분석 데이터",
            },
            "menus": {
                "retriever": menus_retriever,
                "description": "식당명 및 메뉴 및 금액",
            },
            "visit": {
                "retriever": visit_retriever,
                "description": "관광지 핵심 워드 및 정보",
            },
            "kakaomap_reviews": {
                "retriever": kakaomap_reviews_retriever,
                "description": "리뷰 데이터",
            },
        }

        # 각 리트리버의 설명을 임베딩합니다.
        retriever_embeddings = {
            key: embedding.embed_query(info["description"])
            for key, info in retrievers_info.items()
        }

        # 코사인 유사도 계산
        similarities = {
            key: util.cos_sim(input_embedding, embed).item()
            for key, embed in retriever_embeddings.items()
        }

        # 유사도가 임계값 이상인 리트리버 선택
        similarity_threshold = CONFIG['similarity_threshold']
        selected_retrievers = sorted(
            [(key, sim) for key, sim in similarities.items() if sim > similarity_threshold],
            key=lambda x: x[1],
            reverse=True,
        )

        # 선택된 리트리버를 사용해 문서 검색 수행
        results = []
        for retriever_key, _ in selected_retrievers:
            try:
                retriever = retrievers_info[retriever_key]["retriever"]
                result = retriever.get_relevant_documents(query)
                results.extend(result)
            except Exception as e:
                print(f"{retriever_key} 리트리버에서 오류 발생: {e}")
                continue

        return results if results else "관련된 정보가 없습니다."

    except Exception as e:
        print(f"오류 발생: {e}")
        return "오류가 발생했습니다."

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]

# 그 다음 채팅 입력 처리 코드
if prompt := st.chat_input("메시지를 입력하세요", key="user_input"):  # := 연산자 사용
    with st.spinner("🤔 생각하는 중..."):
        try:
            enhanced_prompt = f"""
                사용자 입력: {prompt}
                키워드: {st.session_state.keywords if 'keywords' in st.session_state else '없음'}
                지역: {st.session_state.locations if 'locations' in st.session_state else '없음'}
                최소 평점: {st.session_state.score if 'score' in st.session_state else '없음'}
            """.strip()
            
            response = get_chatbot_response(enhanced_prompt, memory, chain)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant", avatar="🍊"):
                st.markdown(response)
        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.error("응답을 생성하는 과정에서 오류가 발생했습니다. 다시 시도해주세요.")

# 텍스트 표시
st.write("")

# 메시지 표시 - 한 번만 실행
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🐬"):
        st.write(message["content"])

# 사이드바에 초기화 버튼 추가
st.sidebar.button("대화 초기화", on_click=clear_chat_history)

# @st.cache_resource 데코레이터를 수정하고 TTL 추가
@st.cache_resource(ttl=3600)  # 1시간 캐시
def initialize_databases():
    try:
        # JSON 데이터 로드
        data = load_json_files(file_paths)
        
        # 진행 상황을 보여주는 progress bar 추가
        progress_text = "데이터베이스 초기화 중..."
        my_bar = st.progress(0, text=progress_text)
        
        # Document 객체 생성 및 FAISS DB 초기화를 단계별로 진행
        dbs = {}
        total_steps = 7  # 총 처리해야 할 DB 수
        
        # 각 데이터셋 처리를 함수로 분리
        def process_dataset(data_key, data_items, step):
            docs = [Document(page_content=item.get("가게명" if "가게" in data_key else "관광지명", ""), 
                           metadata=item) for item in data_items]
            db = FAISS.from_documents(documents=docs, embedding=embedding)
            my_bar.progress((step + 1) / total_steps, 
                          text=f"{progress_text} ({step + 1}/{total_steps})")
            return db
        
        # 각 데이터셋 순차적 처리
        datasets = [
            ("mct", data["mct"]),
            ("month", data["month"]),
            ("wkday", data["wkday"]),
            ("mop_sentiment", data["mop_sentiment"]),
            ("menu", data["menu"]),
            ("visit_jeju", data["visit_jeju"]),
            ("kakaomap_reviews", data["kakaomap_reviews"])
        ]
        
        for i, (key, items) in enumerate(datasets):
            dbs[f"{key}_db"] = process_dataset(key, items, i)
        
        my_bar.empty()  # progress bar 제거
        return dbs
        
    except Exception as e:
        st.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        return None

# 챗봇 시작 시 이전 대화 기록 불러오기
def main():
    try:
        with st.spinner("데이터베이스 초기화 중..."):
            # 데이터베이스 초기화
            dbs = initialize_databases()
            if dbs is None:
                st.error("데이터베이스 초기화 실패")
                return
            # Retriever 초기화
            global mct_retriever, month_retriever, wkday_retriever, mop_retriever
            global menus_retriever, visit_retriever, kakaomap_reviews_retriever
            
            # BM25 검색기 생성
            mct_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in mct_docs])
            month_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in month_docs])
            wkday_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in wkday_docs])
            mop_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in mop_docs])
            menus_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in menu_docs])
            visit_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in visit_docs])
            kakaomap_reviews_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in kakaomap_reviews_docs])
            
            
            # Retriever 초기화
            mct_retriever = dbs["mct_db"].as_retriever()
            month_retriever = dbs["month_db"].as_retriever()
            wkday_retriever = dbs["wkday_db"].as_retriever()
            mop_retriever = dbs["mop_db"].as_retriever()
            menus_retriever = dbs["menus_db"].as_retriever()
            visit_retriever = dbs["visit_db"].as_retriever()
            kakaomap_reviews_retriever = dbs["kakaomap_reviews_db"].as_retriever()    
    
    except Exception as e:
        st.error(f"초기화 중 오류 발생: {e}")
    
    
# 메인 함수 실행
if __name__ == "__main__":
    main()