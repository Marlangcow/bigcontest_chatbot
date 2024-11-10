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

# .env 파일 경로 지정
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY_1")

# Streamlit App UI
st.set_page_config(page_title="🍊감귤톡")

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

# 텍스트 표시
st.write("")

# Replicate Credentials
with st.sidebar:
    st.title("🍊감귤톡이 다 찾아줄게🍊")

    st.write("")

    st.subheader("원하는 #키워드를 골라봐")

    # selectbox 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    keywords = st.sidebar.selectbox(
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

    st.write("")

    st.subheader("어떤 장소가 궁금해?")

    # 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    locations = st.sidebar.selectbox(
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
    )
    st.write("")

    st.subheader("평점 몇점 이상을 원해?")
    score = st.slider("리뷰 평점", min_value=3.0, max_value=5.0, value=4.5, step=0.05)

# 이메일 링크
st.sidebar.caption("📨 감귤톡에 문의하기 [Send email](mailto:happily2bus@gmail.com)")

st.write("")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🐬"):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]


st.sidebar.button("대화 초기화", on_click=clear_chat_history)

# RAG

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is {device}.")


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

@st.cache_data
# JSON 파일 로드
data = load_json_files(file_paths)

# FAISS 인덱스 경로 설정
index_paths = {
    "mct": "/Users/naeun/bigcontest_chatbot/data/faiss_index/mct_index_pq.faiss",
    "month": "/Users/naeun/bigcontest_chatbot/data/faiss_index/month_index_pq.faiss",
    "wkday": "/Users/naeun/bigcontest_chatbot/data/faiss_index/wkday_index_pq.faiss",
    # "mop": "/Users/naeun/bigcontest_chatbot/data/faiss_index/mop_db.faiss",  # 주석 처리된 경로
    "menu": "/Users/naeun/bigcontest_chatbot/data/faiss_index/menu.faiss",
    "visit": "/Users/naeun/bigcontest_chatbot/data/faiss_index/visit_jeju.faiss",
    "kakaomap_reviews": "/Users/naeun/bigcontest_chatbot/data/faiss_index/kakaomap_reviews.faiss",
}

@st.cache_data
# FAISS 인덱스 로드
faiss_indexes = load_faiss_indexes(index_paths)

# Document 객체 생성
mct_docs = [
    Document(page_content=item.get("가게명", ""), metadata=item) for item in data["mct"]
]
month_docs = [
    Document(page_content=item.get("관광지명", ""), metadata=item)
    for item in data["month"]
]
wkday_docs = [
    Document(page_content=item.get("관광지명", ""), metadata=item)
    for item in data["wkday"]
]
mop_docs = [
    Document(page_content=item.get("관광지명", ""), metadata=item)
    for item in data["mop_sentiment"]
]
menu_docs = [
    Document(page_content=item.get("가게명", ""), metadata=item)
    for item in data["menu"]
]
visit_docs = [
    Document(page_content=item.get("관광지명", ""), metadata=item)
    for item in data["visit_jeju"]
]
kakaomap_reviews_docs = [
    Document(page_content=item.get("관광지명", ""), metadata=item)
    for item in data["kakaomap_reviews"]
]

# 모델과 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)


# HuggingFaceEmbeddings 객체 초기화
embedding = HuggingFaceEmbeddings(model_name=model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

@st.cache_data
# 각 문서 리스트를 FAISS DB에 넣기
mct_db = FAISS.from_documents(documents=mct_docs, embedding=embedding)
month_db = FAISS.from_documents(documents=month_docs, embedding=embedding)
wkday_db = FAISS.from_documents(documents=wkday_docs, embedding=embedding)
mop_db = FAISS.from_documents(documents=mop_docs, embedding=embedding)
menus_db = FAISS.from_documents(documents=menu_docs, embedding=embedding)
visit_db = FAISS.from_documents(documents=visit_docs, embedding=embedding)
kakaomap_reviews_db = FAISS.from_documents(
    documents=kakaomap_reviews_docs, embedding=embedding
)

@st.cache_data
# 데이터베이스를 검색기로 사용하기 위해 retriever 변수에 할당
mct_retriever = mct_db.as_retriever()
month_retriever = month_db.as_retriever()
wkday_retriever = wkday_db.as_retriever()
mop_retriever = mop_db.as_retriever()
menus_retriever = menus_db.as_retriever()
visit_retriever = visit_db.as_retriever()
kakaomap_reviews_retriever = kakaomap_reviews_db.as_retriever()


def initialize_retriever(
    db, search_type="mmr", k=4, fetch_k=10, lambda_mult=0.6, score_threshold=0.6
):
    return db.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "score_threshold": score_threshold,
        },
    )


# 리스트로 DB와 이름을 묶어서 처리
dbs = {
    "mct": mct_db,
    "month": month_db,
    "wkday": wkday_db,
    "mop": mop_db,
    "menus": menus_db,
    "visit": visit_db,
    "kakaomap_reviews": kakaomap_reviews_db,
}

# 각 DB에 대해 리트리버 초기화
retrievers = {name: initialize_retriever(db) for name, db in dbs.items()}

@st.cache_data
# BM25 검색기 생성
mct_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in mct_docs])
month_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in month_docs]
)
wkday_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in wkday_docs]
)
mop_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in mop_docs])
menus_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in menu_docs])
visit_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in visit_docs]
)
kakaomap_reviews_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in kakaomap_reviews_docs]
)


def initialize_ensemble_retriever(retrievers, weights):
    return EnsembleRetriever(retrievers=retrievers, weights=weights)


# 각 DB에 대해 리트리버와 BM25 리트리버 리스트를 묶은 딕셔너리
ensemble_retrievers = {
    "mct": (mct_retriever, mct_bm25_retriever),
    "month": (month_retriever, month_bm25_retriever),
    "wkday": (wkday_retriever, wkday_bm25_retriever),
    "mop": (mop_retriever, mop_bm25_retriever),
    "menus": (menus_retriever, menus_bm25_retriever),
    "visit": (visit_retriever, visit_bm25_retriever),
    "kakaomap_reviews": (kakaomap_reviews_retriever, kakaomap_reviews_bm25_retriever),
}

# Ensemble retrievers 초기화를 명확하게
ensemble_retrievers = {
    name: initialize_ensemble_retriever(
        retrievers=ensemble_retrievers[name],
        weights=[0.6, 0.4],
    )
    for name in ensemble_retrievers.keys()
}

@st.cache_data
def flexible_function_call_search(query):
    try:
        # 입력 쿼리의 임베딩을 가져옵니다.
        input_embedding = embedding.embed_query(query)

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
                "description": "관광지 핵심 키워드 및 정보",
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
        similarity_threshold = 0.7
        selected_retrievers = sorted(
            [
                (key, sim)
                for key, sim in similarities.items()
                if sim > similarity_threshold
            ],
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


def generate_response_with_faiss(
    question,
    df,
    embeddings,
    model,
    embed_text,
    keywords,
    local,
    locations={
        "구좌": "구좌",
        "대정": "대정",
        "안덕": "안덕",
        "우도": "우도",
        "애월": "애월",
        "조천": "조천",
        "제주시내": "제주시내",
        "추자": "추자",
        "한림": "한림",
        "한경": "한경",
    },
    score=0,
    index_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "faiss_index.faiss"
    ),
    max_count=10,
    k=3,
    print_prompt=True,
):
    # FAISS 인덱스 로드
    try:
        index = load_faiss_indexes(index_path)
    except Exception as e:
        return f"인덱스 로드에 실패했습니다: {e}"

    # 검색 쿼리 임베딩 생성
    query_embedding = embed_text(question).reshape(1, -1)

    # 유사한 텍스트 검색
    distances, indices = index.search(query_embedding, k * 3)
    filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True)

    # 키워드 매핑
    keyword_map = {
        "착한가격업소": "착한가격업소",
        "럭셔리트래블인제주": "럭셔리트래블인제주",
        "우수관광사업체": "우수관광사업체",
        "무장애관광": "무장애관광",
        "안전여행스탬프": "안전여행스탬프",
        "향토음식": "향토음식",
        "한식": "한식",
        "카페": "카페",
        "해물뚝배기": "해물뚝배기",
        "몸국": "몸국",
        "해장국": "해장국",
        "수제버거": "수제버거",
        "흑돼지": "흑돼지",
        "해산물": "해산물",
        "일식": "일식",
    }

    # 키워드 필터링
    if keywords in keyword_map:
        filtered_df = filtered_df[
            filtered_df["핵심키워드"].apply(lambda x: keyword_map[keywords] in x)
        ].reset_index(drop=True)
    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    # 지역 필터링
    local = locations.get(local, "기타")
    filtered_df = filtered_df[filtered_df["지역"] == local].reset_index(drop=True)
    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    # 평점 필터링
    filtered_df = filtered_df[filtered_df["평점"] >= score].reset_index(drop=True)
    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    # 프롬프트 생성
    reference_info = "\n".join(filtered_df["text"][:max_count])
    prompt = (
        f"질문: {question} 특히 {local}을 선호해\n참고할 정보:\n{reference_info}\n응답:"
    )
    if print_prompt:
        print("-----------------------------" * 3)
        print(prompt)
        print("-----------------------------" * 3)

    # 응답 생성
    response = model.generate_content(prompt)
    return response

@st.cache_data
# Google Generative AI API 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,  # 더 낮은 temperature로 설정해 할루시네이션 줄임
    top_p=0.85,  # top_p를 조정해 더 예측 가능한 답변 생성
    frequency_penalty=0.1,  # 같은 단어의 반복을 줄이기 위해 패널티 추가
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@st.cache_data
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
    8. 답변 내용에 따라 폰트사이즈, 불렛, 순서를 활용하고 문단을 구분하여 가독성이 좋게 해주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """,
)

# 체인 생성
chain = LLMChain(
    prompt=prompt_template,
    llm=llm,
    output_parser=StrOutputParser(),
)


# 통합된 응답 생성 함수
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

        # 대화 기록 불러오기
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


# # 챗봇 대화 루프 함수 (콘솔용)
# def chat():
#     print("챗봇 대화를 시작합니다. 'exit'을 입력하면 종료됩니다.")
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     while True:
#         query = input("질문을 입력하세요: ")
#         if query.lower() == "exit":
#             print("챗봇을 종료합니다.")
#             break

#         # 통합된 응답 생성 함수 사용
#         response = get_chatbot_response(query, memory, chain)
#         print("\n챗봇 응답:", response)

# 세션 상태 체크 및 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # 메시지 초기화

# 사용자로부터 입력 받기, label 추가
prompt = st.chat_input(
    "Say something", label="Chat Input", label_visibility="collapsed"
)
if prompt:
    # 입력받은 쿼리로 응답 생성
    response = get_chatbot_response(prompt, memory, chain)

    # 입력 및 응답 결과를 UI에 출력
    st.write(f"User: {prompt}")
    st.write(f"Chatbot: {response}")

# 챗봇 시작 시 이전 대화 기록 불러오기
if __name__ == "__main__":
    st.session_state
