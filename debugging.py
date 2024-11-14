import streamlit as st
import os
from src.chatbot import get_chatbot_response
from src.models import initialize_llm, create_chain
from src.data_loader import (
    create_documents,
    load_faiss_indexes_with_retriever,
    initialize_embeddings,
)
from src.config import INDEX_PATHS, PKL_PATHS, INDEX_PATHS
from src.retrievers import load_retrievers_from_json
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import streamlit as st
import json
import multiprocessing

st.set_page_config(
    page_title="감귤톡",
    page_icon="🍊",
    layout="wide",
)
from src.config import *
from src.ui import (
    initialize_streamlit_ui,
    display_main_image,
    setup_sidebar,
    setup_keyword_selection,
    setup_location_selection,
    setup_score_selection,
    clear_chat_history,
)
from src.data_loader import *
from src.models import *
from src.retrievers import *
from src.chatbot import *
from src.prompts import get_chat_prompt

from langchain.memory import ConversationBufferMemory

import gzip


# Streamlit UI 초기화
def initialize_streamlit_ui():
    # st.session_state.messages 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        ]

    # 제목 및 정보 텍스트 설정
    st.title("🍊감귤톡, 제주도 여행 메이트")
    st.write("")
    st.info("제주도 여행 메이트 감귤톡이 제주도의 방방곡곡을 알려줄게 🏝️")

    # 이미지 표시
    display_main_image()

    # 메시지 표시
    display_messages()

    with st.sidebar:
        setup_sidebar()


def display_main_image():
    image_path = "https://img4.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202105/25/linkagelab/20210525013157546odxh.jpg"
    st.image(image_path, use_container_width=False)
    st.write("")


def setup_sidebar():
    st.title("🍊감귤톡이 다 찾아줄게🍊")
    st.write("")
    setup_keyword_selection()
    setup_location_selection()
    setup_score_selection()
    st.button("대화 초기화", on_click=clear_chat_history)
    st.write("")
    st.caption("📨 감귤톡에 문의하기 [Send email](mailto:happily2bus@gmail.com)")


def setup_keyword_selection():
    st.subheader("원하는 #키워드를 골라봐")
    keywords = st.selectbox(
        "키워드 선택",
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
        label_visibility="collapsed",
    )
    st.write("")


def setup_location_selection():
    st.subheader("어떤 장소가 궁금해?")
    locations = st.selectbox(
        "장소 선택",
        [
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
        ],
        key="locations",
        label_visibility="collapsed",
    )
    st.write("")


def setup_score_selection():
    st.subheader("평점 몇점 이상을 원해?")
    score = st.slider("리뷰 평점", min_value=3.0, max_value=5.0, value=4.5, step=0.05)
    st.write("")


def display_messages():
    for message in st.session_state.messages:
        role = "🐬" if message["role"] == "assistant" else "👤"
        st.write(f"{role} {message['content']}")
    # 메시지가 없을 경우 기본 메시지 추가
    if not st.session_state.messages:
        st.session_state.messages.append(
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        )


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]


# 메인 실행 코드
def main():

    # 데이터 로드 및 임베딩 초기화
    data = load_retrievers_from_json()  # JSON 파일에서 데이터 로드
    embedding_function = initialize_embeddings()

    # FAISS 인덱스 및 리트리버 로드
    retrievers = load_faiss_indexes_with_retriever(
        INDEX_PATHS, data, embedding_function
    )

    # 세션 상태에서 retrievers 키가 없는 경우 초기화
    if "retrievers" not in st.session_state:
        st.session_state.retrievers = retrievers

    # 세션 상태에서 memory 키가 없는 경우 초기화
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    # Streamlit UI 초기화
    initialize_streamlit_ui()

    # 사용자 입력 처리
    if prompt := st.chat_input("무엇이 궁금하신가요?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 챗봇 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하고 있습니다..."):
                response = get_chatbot_response(
                    user_input=prompt,
                    memory=st.session_state.memory,
                    chain=st.session_state.chain,
                    retrievers=st.session_state.retrievers,
                )
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main()
