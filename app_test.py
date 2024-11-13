from src.config import *
from src.data_loader import *
from src.models import *
from src.retrievers import *
from src.chatbot import *
from src.prompts import get_chat_prompt
from src.ui import (
    initialize_streamlit_ui,
    display_main_image,
    setup_sidebar,
    setup_keyword_selection,
    setup_location_selection,
    setup_score_selection,
    clear_chat_history,
)
from langchain.memory import ConversationBufferMemory
import streamlit as st
import gzip
import pickle

# Google API 키 불러오기
google_api_key = st.secrets["google_api_key"]


def manage_chat_history():
    """채팅 히스토리 관리 함수"""
    if len(st.session_state.messages) > st.session_state.max_messages:
        # 가장 오래된 메시지 제거 (처음 2개는 시스템 메시지로 보존)
        st.session_state.messages = (
            st.session_state.messages[:2]
            + st.session_state.messages[-(st.session_state.max_messages - 2) :]
        )

        # 메모리도 함께 정리
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        if len(chat_history) > st.session_state.max_messages:
            st.session_state.memory.clear()
            # 최근 대화만 다시 저장
            for msg in st.session_state.messages[2:]:  # 시스템 메시지 제외
                if msg["role"] == "user":
                    st.session_state.memory.save_context(
                        {"input": msg["content"]}, {"output": ""}
                    )
                elif msg["role"] == "assistant":
                    st.session_state.memory.save_context(
                        {"input": ""}, {"output": msg["content"]}
                    )


class DocumentSearcher:
    def __init__(self, retriever_path: str):
        """
        문서 검색기 초기화

        Args:
            retriever_path (str): 앙상블 리트리버가 저장된 파일 경로
        """
        self.ensemble_retrievers = load_ensemble_retrievers(retriever_path)

    def search(self, query: str, category: str, top_k: int = 3):
        """
        사용자 쿼리에 따른 문서 검색

        Args:
            query (str): 사용자 검색어
            category (str): 검색할 문서 카테고리
            top_k (int): 반환할 문서 개수

        Returns:
            list: 관련 문서 리스트
        """
        try:
            if category not in self.ensemble_retrievers:
                raise ValueError(f"유효하지 않은 카테고리입니다: {category}")

            retriever = self.ensemble_retrievers[category]
            results = retriever.get_relevant_documents(query)
            return results[:top_k]

        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            return []


def handle_streamlit_input(chain, memory, prompt):
    try:
        # 디버깅을 위한 로그 추가
        st.write("사용자 입력:", prompt)
        chat_history = memory.load_memory_variables({})["chat_history"]
        st.write("대화 기록:", chat_history)

        # Chain 실행
        response = chain(
            {
                "user_input": prompt,
                "chat_history": chat_history,
                "keyword": st.session_state.get("keywords", ""),
                "location": st.session_state.get("locations", ""),
                "min_score": st.session_state.get("score", 4.5),
                "search_results": "search_results",
            }
        )

        # 응답 처리
        st.markdown(response["output"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response["output"]}
        )
        manage_chat_history()  # 히스토리 관리 함수 호출

        # 메모리 업데이트
        memory.save_context({"input": prompt}, {"output": response["output"]})

    except Exception as e:
        st.error(f"응답 생성 중 오류 발생: {str(e)}")
        print(f"오류 상세: {str(e)}")  # 디버깅용


def main():
    initialize_streamlit_ui()

    # st.session_state 변수 초기화
    if "memory" not in st.session_state:
        # ConversationBufferMemory를 사용하여 메모리 초기화
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        ]

    # 필요한 구성 요소를 초기화
    if "chain" not in st.session_state:
        llm = initialize_llm()  # LLM 초기화
        prompt_template = get_chat_prompt()  # 프롬프트 템플릿 가져오기
        st.session_state.chain = create_chain(
            llm, prompt_template, memory=st.session_state.memory
        )

    # retrievers 불러오기
    retrievers = load_retrievers_from_pkl(file_path)
    if retrievers:
        st.session_state.retrievers = retrievers
        st.write("retrievers 데이터가 세션에 로드되었습니다.")
    else:
        st.write("retrievers 데이터 로드 실패")

    # 이후 retrievers를 사용하는 코드 처리 (예: 사용자 질의 처리 등)
    if "retrievers" in st.session_state:
        st.write("retrievers 데이터가 세션에 로드되었습니다.")
    else:
        st.write("retrievers 데이터가 세션에 로드되지 않았습니다.")

    # 이전 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("무엇이 궁금하신가요?"):
        # 사용자 메시지 표시
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
