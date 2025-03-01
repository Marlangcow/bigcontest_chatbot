import os
import google.generativeai as genai
import streamlit as st
import faiss

# API 키 설정
GOOGLE_API_KEY = st.secrets["google_api_key"]

# 파일 경로 설정
MODULE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# FAISS 인덱스 경로
INDEX_PATHS = {
    "mct_db_index_cpu": "./data/faiss_db_index_cpu/mct_db_index_cpu.faiss",
    "month_db_index_cpu": "./data/faiss_db_index_cpu/month_db_index_cpu.faiss",
    "wkday_db_index_cpu": "./data/faiss_db_index_cpu/wkday_db_index_cpu.faiss",
    "mob_db_index_cpu": "./data/faiss_db_index_cpu/mob_db_index_cpu.faiss",
    "menus_db_index_cpu": "./data/faiss_db_index_cpu/menus_db_index_cpu.faiss",
    "visit_db_index_cpu": "./data/faiss_db_index_cpu/visit_db_index_cpu.faiss",
    "kakaomap_reviews_index_cpu": "./data/faiss_db_index_cpu/kakaomap_reviews_index_cpu.faiss",
}

# JSON 파일 경로
JSON_PATHS = {
    # json 리트리버 파일 경로
    "mct_json": "./data/mct.json",
    "month_json": "./data/month.json",
    "wkday_json": "./data/wkday.json",
    "mop_sentiment_json": "./data/merge_mop_sentiment.json",
    "menu_json": "./data/mct_menus.json",
    "visit_jeju_json": "./data/visit_jeju.json",
    "kakaomap_reviews_json": "./data/kakaomap_reviews.json",
}

# 위치 매핑
LOCATIONS = {
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
}

# 키워드 매핑
KEYWORD_MAP = {
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

# 임베딩 모델 이름
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# 최대 토큰 수
MAX_TOKENS = 5000

# 온도
TEMPERATURE = 0.2

# Top-p
TOP_P = 0.85
