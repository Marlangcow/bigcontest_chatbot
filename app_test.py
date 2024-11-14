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

# Google API 키 불러오기
google_api_key = st.secrets["google_api_key"]

# .json 파일 경로 가져오기
retriever_file_paths = glob.glob(
    "/Users/naeun/bigcontest_chatbot/data/json_retrievers/*.json"
)


# 리트리버 데이터 로드 (병렬화 적용)
def load_retrievers_parallel(file_paths):
    # 멀티프로세싱을 사용하여 리트리버 파일을 병렬로 로드합니다.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        retriever_data = pool.map(load_ensemble_retriever_from_json, file_paths)
    return retriever_data


# 채팅 기록 관리 함수
def manage_chat_history():
    if len(st.session_state.messages) > st.session_state.max_messages:
        st.session_state.messages = (
            st.session_state.messages[:2]
            + st.session_state.messages[-(st.session_state.max_messages - 2) :]
        )
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        if len(chat_history) > st.session_state.max_messages:
            st.session_state.memory.clear()
            for msg in st.session_state.messages[2:]:
                if msg["role"] == "user":
                    st.session_state.memory.save_context(
                        {"input": msg["content"]}, {"output": ""}
                    )
                elif msg["role"] == "assistant":
                    st.session_state.memory.save_context(
                        {"input": ""}, {"output": msg["content"]}
                    )


# 리트리버 데이터 로드
def load_retrievers():
    if "retriever_data" not in st.session_state:
        # retrievers.py에서 정의한 함수 호출하여 데이터 로드
        retriever_data = load_ensemble_retriever_from_json(retriever_file_paths)
        if retriever_data:
            st.session_state.retriever_data = retriever_data
            st.session_state.retrievers = retriever_data  # "retrievers" 키 초기화
            st.write("모든 JSON 리트리버가 성공적으로 로드되었습니다!")
        else:
            st.write("리트리버 로드 실패")
    else:
        retriever_data = st.session_state.retriever_data
    return retriever_data


# 리트리버 로드
retriever_data = load_retrievers()

# # 리트리버 데이터가 로드되었다면 사용
# if retriever_data:
#     for key, ensemble_retriever in retriever_data.items():
#         st.write(f"앙상블 리트리버 ({key}):", ensemble_retriever)


# Streamlit main function
def main():
    # Streamlit UI 초기화
    initialize_streamlit_ui()

    # 세션 상태 변수 초기화
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    # 초기 메시지 표시
    if "initialized" not in st.session_state:
        st.chat_message("assistant").markdown("어떤 곳을 찾아줄까?")
        st.session_state.initialized = True

    # 세션 상태 변수 초기화
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        ]

    # chain 및 retrievers 초기화
    if "chain" not in st.session_state:
        llm = initialize_llm()
        prompt_template = get_chat_prompt()
        st.session_state.chain = create_chain(
            llm, prompt_template, memory=st.session_state.memory
        )

    # 리트리버 로드
    retrievers = load_retrievers()

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
