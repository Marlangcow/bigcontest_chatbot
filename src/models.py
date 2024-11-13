from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from .config import GOOGLE_API_KEY
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from functools import wraps
from transformers import AutoTokenizer
import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler


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


# 모델에 맞는 토크나이저 불러오기 (gemini 모델에 맞는 토크나이저 사용)
tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")


# 토큰 수 계산 함수
def count_tokens(text: str) -> int:
    """주어진 텍스트의 토큰 수를 계산하는 함수"""
    return len(tokenizer.encode(text))


# 토큰 수 제한을 위한 함수
def limit_input_tokens(text: str, max_tokens: int) -> str:
    """입력 텍스트의 토큰 수가 max_tokens를 초과하지 않도록 제한"""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # 토큰 수가 초과되면 잘라냄
    return tokenizer.decode(tokens)


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
    # Streamlit UI에서 받은 인자값들 가져오기
    keywords = st.session_state.get("keywords", "")
    locations = st.session_state.get("locations", "")
    score = st.session_state.get("score", 4.5)
    user_input = st.session_state.get("user_input", "")
    chat_history = st.session_state.get("chat_history", [])
    search_results = st.session_state.get("search_results", [])

    # 'input_text' 변수를 추가하여 prompt에 전달
    input_text = user_input  # user_input을 input_text로 저장

    # 입력 텍스트의 토큰 수 확인 및 제한
    prompt = prompt_template.template.format(
        keyword=keywords,
        location=locations,
        min_score=score,
        user_input=user_input,
        chat_history=chat_history,
        search_results=search_results,
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


def get_embedding_model():
    """
    한국어 SRoBERTa 임베딩 모델을 초기화하고 반환하는 함수
    """
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {
        "device": "cpu",
        "trust_remote_code": True,
    }
    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": 32,
        "max_length": 512,  # 임베딩 입력 길이 제한
    }

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./models",
    )
