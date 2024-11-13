import os
import google.generativeai as genai
import streamlit as st

# API 키 설정
GOOGLE_API_KEY = st.secrets["google_api_key"]

# 파일 경로 설정
MODULE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JSON 파일 경로
FILE_PATHS = {
    "mct": "/Users/naeun/bigcontest_chatbot/data/mct.json",
    "month": "/Users/naeun/bigcontest_chatbot/data/month.json",
    "wkday": "/Users/naeun/bigcontest_chatbot/data/wkday.json",
    "mop_sentiment": "/Users/naeun/bigcontest_chatbot/data/merge_mop_sentiment.json",
    "menu": "/Users/naeun/bigcontest_chatbot/data/mct_menus.json",
    "visit_jeju": "/Users/naeun/bigcontest_chatbot/data/visit_jeju.json",
    "kakaomap_reviews": "/Users/naeun/bigcontest_chatbot/data/kakaomap_reviews.json",
}

# FAISS 인덱스 경로
INDEX_PATHS = {
    "mct": "/Users/naeun/bigcontest_chatbot/data/faiss_index/mct_index_pq.faiss",
    "month": "/Users/naeun/bigcontest_chatbot/data/faiss_index/month_index_pq.faiss",
    "wkday": "/Users/naeun/bigcontest_chatbot/data/faiss_index/wkday_index_pq.faiss",
    "mop": "/Users/naeun/bigcontest_chatbot/data/faiss_index/mop_flat_l2.faiss",
    "menu": "/Users/naeun/bigcontest_chatbot/data/faiss_index/menu.faiss",
    "visit": "/Users/naeun/bigcontest_chatbot/data/faiss_index/visit_jeju.faiss",
    "kakaomap_reviews": "/Users/naeun/bigcontest_chatbot/data/faiss_index/kakaomap_reviews.faiss",
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
