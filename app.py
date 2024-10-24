import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st
import langchain.chat_models

from transformers import AutoTokenizer, AutoModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
# from langchain.chains import ConversationChain
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI

import torch
import faiss

# .env 파일 경로 지정
google_api_key = os.getenv("GOOGLE_API_KEY")


# CSV 파일과 .npy 파일 경로 설정
csv_file_paths = [
    './data/review_documents.csv',
    './data/mct_documents.csv',
    './data/trrsrt_documents.csv'
]
dfs = [pd.read_csv(csv_file_path) for csv_file_path in csv_file_paths]

# FAISS 인덱스 파일 경로
faiss_index_path = './modules/faiss_index.index'

# FAISS 인덱스 로드
faiss_index = faiss.read_index(faiss_index_path)

# 임베딩 모델 로드 (예: 'jhgan/ko-sroberta-multitask')
model_embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')


# Google Generative AI API 설정
chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                                    api_key='AIzaSyAnl8_XMJ-rJgZ4mGBqsUuq8A4jGESxPAo',
                                    temperature=0.3,  
                                    top_p=0.85,       
                                    frequency_penalty=0.1  
)

# 멀티턴 대화를 위한 Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 멀티턴 프롬프트 템플릿 설정 (COT 방식 적용)
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""..."""  # 원본 템플릿 내용 그대로 사용
)

# 검색 및 응답 생성 함수
def search_faiss(query_embedding, k=5):
    # FAISS에서 유사한 벡터를 검색하여 원본 데이터 반환
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)
    search_results = []
    total_length = 0

    for idx in indices[0]:
        found = False
        for df in dfs:
            if total_length + len(df) > idx:
                if idx - total_length >= 0 and idx - total_length < len(df):
                    search_results.append(df.iloc[idx - total_length])
                found = True
                break
            total_length += len(df)
        if found:
            continue

    return search_results

# 대화형 응답 생성 함수
def generate_response(user_input):
    query_embedding = model_embedding.encode([user_input])
    search_results = search_faiss(query_embedding)
    search_results_str = "\n".join([result.to_string() for result in search_results])
    
    filled_prompt = prompt_template.format(
        input_text=user_input,
        search_results=search_results_str,
        chat_history=memory.load_memory_variables({})["chat_history"]
    )

    response_parts = []
    while filled_prompt:
        part = filled_prompt[:5000]
        filled_prompt = filled_prompt[5000:]

        response = chat_model.invoke([{"role": "user", "content": part}])
        response_parts.append(response.content)

        if len(response_parts) >= 3:
            break

    for part in response_parts:
        memory.save_context({"input": user_input}, {"output": part})

    return "\n".join(response_parts)

# Streamlit 페이지 설정
st.set_page_config(page_title="jeju-chatbot", page_icon="🌴")
st.title("🌴 🌴 제주도 여행 메이트 AI")

# 사용자 입력을 위한 텍스트 입력창
st.write("제주도 특급 가이드! 맛집부터 카페, 관광지까지 원하는 곳을 말해봐!")
message = st.text_input("찾고 있는 장소의 특징을 알려줘.", key="input")

# 대화 기록을 저장할 리스트
if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button("전송"):
    if message:
        try:
            # 모델 호출 및 응답 받기
            response = generate_response(message)
            # 대화 기록에 추가
            st.session_state['history'].append({"user": message, "bot": response})
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

# 대화 기록을 화면에 출력
if st.session_state['history']:
    for chat in st.session_state['history']:
        st.write(f"**사용자**: {chat['user']}")
        st.write(f"**AI**: {chat['bot']}")
