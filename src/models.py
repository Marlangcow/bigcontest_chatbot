from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler
from .config import GOOGLE_API_KEY
from typing import Optional
from functools import wraps
import streamlit as st


class MaxLLMCallsExceeded(Exception):
    """LLM 최대 호출 횟수 초과 예외"""

    pass


def limit_llm_calls(func):
    """LLM 호출 횟수를 제한하는 데코레이터"""
    call_count = 0
    max_calls = 3

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal call_count
        if call_count >= max_calls:
            raise MaxLLMCallsExceeded("최대 LLM 호출 횟수(3회)를 초과했습니다.")
        call_count += 1
        return func(*args, **kwargs)

    return wrapper


def initialize_llm():
    """LLM을 초기화하는 함수"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        top_p=0.85,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=1000,
        streaming=True,
        max_tokens=5000,
    )


@limit_llm_calls
def create_chain(llm, prompt_template, memory: Optional[dict] = None):
    """
    LLM 체인을 생성하는 함수
    호출 횟수가 제한되며, 최대 입력 토큰이 제한됩니다.
    """
    # session_state 초기화 및 값 체크
    if "keywords" not in st.session_state:
        st.session_state["keywords"] = ""  # 기본값 설정
    if "locations" not in st.session_state:
        st.session_state["locations"] = ""
    if "score" not in st.session_state:
        st.session_state["score"] = 4.5
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    # 'input_text' 변수를 추가하여 prompt에 전달
    input_text = st.session_state["user_input"]  # user_input을 input_text로 저장

    # 입력 텍스트의 토큰 수 확인 및 제한
    prompt = prompt_template.template.format(
        keyword=st.session_state["keywords"],
        location=st.session_state["locations"],
        min_score=st.session_state["score"],
        user_input=st.session_state["user_input"],
        chat_history=st.session_state["chat_history"],
        search_results=st.session_state["search_results"],
        input_text=input_text,  # 추가된 부분
    )

    return LLMChain(
        prompt=prompt_template,
        llm=llm,
        memory=memory,
        output_parser=StrOutputParser(),
        verbose=True,
        callbacks=[StreamingStdOutCallbackHandler()],  # 토큰 사용량 모니터링
    )
