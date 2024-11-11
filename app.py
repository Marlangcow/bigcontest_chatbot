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

# Streamlit 페이지 설정
st.set_page_config(page_title="🍊감귤톡")
st.markdown(STYLES, unsafe_allow_html=True)

# Streamlit App UId
st.title("🍊감귤톡, 제주도 여행 메이트")
st.info("제주도 여행 메이트 감귤톡이 제주도의 방방곡곡을 알려줄게🌴")

# 이미지 로드 설정
if "image_loaded" not in st.session_state:
    st.session_state.image_loaded = True
    st.session_state.image_html = """
    <div style="display: flex; justify-content: center;">
        <img src="https://img4.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202105/25/linkagelab/20210525013157546odxh.jpg" alt="centered image" width="50%">
    </div>
    """

# # 이미지 표시 (세션 상태에서 확인)
# if st.session_state.image_loaded:
#     st.markdown(st.session_state.image_html, unsafe_allow_html=True)
#     # 이미지가 표시된 후 다시 상태를 False로 변경하여 중복 표시 방지
#     st.session_state.image_loaded = False

st.write("")  # 여백 추가

# .env 파일 경로 지정
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


# CSV 파일 로드
@st.cache_data

# CSV 파일 로드
def load_data():
    csv_file_paths = [
        "./data/review_documents.csv",
        "./data/mct_documents.csv",
        "./data/trrsrt_documents.csv",
    ]
    dfs = []

    with st.spinner("잠시만 기다려 주세요. 곧 나와요!"):  # 사용자 정의 스피너 메시지
        dfs = [pd.read_csv(csv_file_path) for csv_file_path in csv_file_paths]

    return dfs


dfs = load_data()


# FAISS 인덱스 파일 경로
faiss_index_path = "./modules/faiss_index.index"

# FAISS 인덱스 로드
faiss_index = faiss.read_index(faiss_index_path)


# 임베딩 모델 로드
@st.cache_data
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")


model_embedding = load_model()


# Google Generative AI API 설정
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=google_api_key,
    temperature=0.3,
    top_p=0.85,
    frequency_penalty=0.3,
)

# 멀티턴 대화를 위한 Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 멀티턴 프롬프트 템플릿 설정 (COT 방식 적용)
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""
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
    8. 문장이 구분되도록 문단을 구분해주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """,
)


# 검색 및 응답 생성 함수
def search_faiss(query_embedding, k=5):
    """
    FAISS에서 유사한 벡터를 검색하여 원본 데이터 반환
    """
    # FAISS 인덱스에서 유사한 벡터 검색
    distances, indices = faiss_index.search(
        np.array(query_embedding, dtype=np.float32), k
    )

    # 검색된 인덱스를 바탕으로 원본 데이터 가져오기
    search_results = []
    total_length = 0  # 전체 길이 초기화

    for idx in indices[0]:
        found = False  # 찾은 데이터프레임 체크
        for df in dfs:
            if (
                total_length + len(df) > idx
            ):  # 현재 데이터프레임에서 유효한 인덱스인지 체크
                if idx - total_length >= 0 and idx - total_length < len(df):
                    search_results.append(df.iloc[idx - total_length])  # 인덱스 재조정
                found = True
                break
            total_length += len(df)  # 전체 길이에 데이터프레임 길이 추가
        if found:  # 이미 찾은 경우 더 이상 반복할 필요 없음
            continue

    return search_results


# 대화형 응답 생성 함수
def generate_response(user_input):
    """
    사용자의 입력을 받아 FAISS 검색 후 응답 생성 (COT 적용)
    """
    # 사용자의 질문을 임베딩으로 변환
    query_embedding = model_embedding.encode([user_input])

    # FAISS 검색 수행
    search_results = search_faiss(query_embedding)

    # 검색된 결과를 텍스트 형식으로 변환
    search_results_str = "\n".join([result.to_string() for result in search_results])

    # PromptTemplate에 검색된 결과와 대화 기록 채우기
    filled_prompt = prompt_template.format(
        input_text=user_input,
        search_results=search_results_str,
        chat_history=memory.load_memory_variables({})["chat_history"],
    )

    # 1회 호출에서 5000 토큰 제한이므로 적절하게 텍스트를 나누어 처리
    response_parts = []
    while filled_prompt:
        # 최대 5000 토큰까지 잘라서 호출
        part = filled_prompt[:5000]
        filled_prompt = filled_prompt[5000:]

        # Google Generative AI API 호출 (대신 사용할 모델로 수정 가능)
        response = chat_model.invoke([{"role": "user", "content": part}])
        response_parts.append(response.content)

        # 호출 횟수 체크
        if len(response_parts) >= 3:
            break  # 최대 3회 호출 제한

    # 메모리에 대화 기록 저장
    for part in response_parts:
        memory.save_context({"input": user_input}, {"output": part})

    # 최종 응답 합치기
    return "\n".join(response_parts)


# 스트림릿 챗봇 인터페이스
if "messages" not in st.session_state:
    st.session_state.messages = []  # messages 초기화

# 이미지 표시 (세션 상태 유지)
st.markdown(st.session_state.image_html, unsafe_allow_html=True)

# 이전 대화 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 마지막 메시지가 어시스턴트의 메시지가 아닐 경우 새 응답 생성
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                placeholder = st.empty()
                full_response = ""  # 응답 초기화

                # 응답을 문자열로 변환
                if isinstance(response, str):
                    full_response = response
                else:
                    full_response = response.text

                # 전체 응답 표시
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
