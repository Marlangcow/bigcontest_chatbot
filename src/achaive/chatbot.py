from langchain.memory import ConversationBufferMemory
from typing import List
from langchain.schema import Document
import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from src.prompts import get_chat_prompt
from src.models import initialize_llm


# 임베딩 모델 초기화
embedding = SentenceTransformer("jhgan/ko-sroberta-multitask")


def initialize_chain():
    # prompt_template 가져오기
    prompt_template = get_chat_prompt()

    llm = initialize_llm()

    if not prompt_template or not llm:
        print("Error: prompt_template 또는 llm이 None입니다.")
        return None

    # 새로운 방식으로 chain 생성
    chain = RunnableSequence(first=prompt_template, last=llm)
    return chain


# chain 객체 초기화
chain = initialize_chain()
if chain is None:
    print("Error: chain 객체가 None입니다.")
else:
    print("chain 객체가 성공적으로 초기화되었습니다.")


# FAISS 검색 및 유사도 계산 함수
def flexible_function_call_search(query, retrievers):
    input_embedding = embedding.encode(query)

    retriever_descriptions = {
        "mct": "식당명 및 이용 비중 및 금액 비중",
        "mct_menus": "식당명 및 메뉴 및 금액",
        "mop": "관광지 전체 키워드 분석 데이터",
        "month": "관광지 월별 조회수",
        "visit": "관광지 핵심 키워드 및 정보",
        "wkday": "주별 일별 조회수 및 연령별 성별별 선호도",
        "kakaomap_reviews": "리뷰 데이터",
    }

    retriever_embeddings = {
        key: torch.tensor(embedding.encode(value), dtype=torch.float32)
        for key, value in retriever_descriptions.items()
    }

    similarities = {
        key: util.cos_sim(input_embedding, embed).item()
        for key, embed in retriever_embeddings.items()
    }

    selected_retrievers = [key for key, sim in similarities.items() if sim >= 0.3]
    if not selected_retrievers:
        selected_retrievers = [max(similarities, key=similarities.get)]

    combined_results = []
    for retriever_key in selected_retrievers:
        retriever = retrievers.get(retriever_key)
        if retriever:
            search_result = retriever.retrieve(query)
            combined_results.extend(search_result)

    unique_results = {doc.page_content: doc for doc in combined_results}.values()

    return list(unique_results)


def get_chatbot_response(user_input, memory, chain, retrievers):
    # 사용자 입력을 기반으로 관련 문서 검색
    search_results = flexible_function_call_search(user_input, retrievers)

    # 검색 결과를 문자열로 변환
    search_results_text = "\n".join([doc.page_content for doc in search_results])

    # Chain에 입력할 변수 준비
    chain_input = {
        "input_text": user_input,
        "user_input": user_input,
        "search_results": search_results_text,
        "chat_history": memory.buffer if memory else [],
        "keyword": st.session_state.get("keywords", "일반"),  # 기본값 설정
        "location": st.session_state.get("locations", "제주시내"),  # 기본값 설정
        "min_score": st.session_state.get("score", 4.5),  # 기본값 설정
    }

    # Chain을 통해 응답 생성
    response = chain.invoke(chain_input)

    # 메모리에 대화 저장
    if memory:
        memory.save_context({"input": user_input}, {"output": response})

    return response


class ChatBot:
    def __init__(self, retrievers):
        self.chain = initialize_chain()
        self.memory = ConversationBufferMemory()
        self.retrievers = retrievers

    def generate_response(self, user_input):
        try:
            if not user_input.strip():
                return "입력이 비어있습니다. 질문을 입력해주세요."

            response = get_chatbot_response(
                user_input, self.memory, self.chain, self.retrievers
            )

            if not response:
                return "응답 생성 중 문제가 발생했습니다. 다시 시도해주세요."

            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "죄송합니다. 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
