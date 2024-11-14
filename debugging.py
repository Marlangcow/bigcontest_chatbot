import toml
import streamlit as st
from src.config import INDEX_PATHS, JSON_PATHS
from src.retrievers import initialize_retrievers
from src.chatbot import get_chatbot_response
from src.ui import (
    initialize_streamlit_ui,  # src/ui.py의 함수를 사용
    display_main_image,
    setup_sidebar,
    setup_location_selection,
    setup_keyword_selection,
    setup_score_selection,
    clear_chat_history,
)
import google.generativeai as genai

GOOGLE_API_KEY = st.secrets["google_api_key"]

genai.configure(api_key=GOOGLE_API_KEY)


def display_messages():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]


# 메인 실행 코드
def main():
    # FAISS 인덱스 및 리트리버 로드
    retrievers = initialize_retrievers(INDEX_PATHS)

    if "retrievers" not in st.session_state:
        st.session_state.retrievers = retrievers

    # if "memory" not in st.session_state:
    #     st.session_state.memory = NewMemoryClass()

    # "chain" 속성 초기화
    if "chain" not in st.session_state:
        from src.chatbot import initialize_chain  # 필요한 경우 import 추가

        st.session_state.chain = initialize_chain()  # chain 객체 초기화

    # UI 초기화 함수 호출
    initialize_streamlit_ui()  # src/ui.py의 함수를 호출

    # 사이드바 설정
    setup_sidebar()

    if prompt := st.chat_input("무엇이 궁금하신가요?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

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
