import os
import numpy as np
import pandas as pd
import streamlit as st

from transformers import AutoTokenizer, AutoModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import faiss

import streamlit as st

# 경로 설정
data_path = '/.data'
module_path = '.modules'

# Gemini 설정
chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CSV 파일과 .npy 파일 경로 설정
csv_file_paths = [
    'FINAL_REVIEW.csv',
    'FINAL_MCT.csv',
    'FINAL_TRRSRT.csv'
]
npy_file_paths = ['V2/all_embeddings_v2.npy']
index_faiss_path = 'combined_db.index'
 
# CSV 파일 로드 (low_memory=False로 DtypeWarning 방지)
dfs = [pd.read_csv(csv_file_path, low_memory=False) for csv_file_path in csv_file_paths]

# .npy 파일 로드
embeddings = np.load(npy_file_paths[0])

# 임베딩 벡터의 차원 확인
dimension = embeddings[0].shape[1] 

# FAISS 인덱스 불러오기
faiss_db = faiss.read_index(index_faiss_path)

# FAISS 인덱스에 저장된 총 벡터 개수 
faiss_db.add(embeddings)


# 임베딩 모델 로드 (예: 'jhgan/ko-sroberta-multitask')
model_embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')


# 멀티턴 대화를 위한 Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 멀티턴 프롬프트 템플릿 설정 (COT 방식 적용)
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""
   ### 역할
    당신은 제주도 맛집과 관광지 추천 전문가입니다. 질문을 받을 때 논리적으로 생각한 후 단계별로 답변을 제공합니다. 복잡한 질문일수록 천천히 생각하고 적절한 데이터를 바탕으로 답변을 제공합니다.

    ### Chain of Thought 방식 적용:
    1. 사용자의 질문을 단계별로 분석합니다.
    2. 먼저 질문의 위치 정보를 파악합니다.
    3. 그 후에 사용자가 제공한 정보나 검색된 데이터를 바탕으로 관련성 있는 맛집과 관광지를 추천합니다.
    4. 단계를 나누어 정보를 체계적으로 제공합니다.

    ### 단계적 사고:
    1. 사용자 질문 분석
    2. 위치 정보 확인
    3. 관련 데이터 검색
    4. 추천 맛집 및 관광지 제공
    5. 추가 질문에 대한 친근한 대화 유지

    ### 지시사항
    당신은 사용자로부터 제주도의 맛집(식당, 카페 등)과 관광지를 추천하는 챗봇입니다.
    1. 사용자가 알고자 하는 동네(시군구)를 알려줄 때 까지 사용자에게 반문하세요. 이는 가장 중요합니다. 단, 위치를 두번 이상 반문하지 마세요. 만약 사용자가 위치를 모른다면 제일 평점이 좋은 3개의 식당+카페와 3개의 관광지를 안내해주세요.
    2. 친근하고 재미있으면서도 정겹게 안내하세요.
    3. source_id는 문서 번호입니다. 따라서 답변을 하는 경우 몇 번 문서를 인용했는지 답변 뒤에 언급하세요.
    4. 추천 할 때, 추천 이유와 소요되는 거리, 평점과 리뷰들도 보여줘. 만약 리뷰가 없는 곳이라면 ("작성된 리뷰가 없습니다.") 라고 해주세요.
    5. 4번의 지시사항과 함께 판매 메뉴 2개, 가격도 알려주세요.
    6. 만약 관광지와 식당이 구글검색에서 나오는 곳이면 지도(map)링크도 같이 첨부해줘. 지도 링크가 없는 곳은 지도 여부를 노출하지 말아주세요.
    7. 실제로 존재하는 식당과 관광지명을 추천해주어야 하며, %%흑돼지 맛집, %%횟집 등의 답변은 하지 말아주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """
)

# 4. 검색 및 응답 생성 함수
def search_faiss(query_embedding, k=5):
    """
    FAISS에서 유사한 벡터를 검색하여 원본 데이터 반환
    """
    # FAISS 인덱스에서 유사한 벡터 검색
    distances, indices = faiss_db.search(np.array(query_embedding, dtype=np.float32), k)

    # 검색된 인덱스를 바탕으로 원본 데이터 가져오기
    search_results = []
    for idx in indices[0]:
        for df in dfs:
            if idx < len(df):
                search_results.append(df.iloc[idx])
                break
            else:
                idx -= len(df)  # 인덱스가 초과되면 다음 데이터셋으로 넘어감

    return search_results

# 5. 대화형 응답 생성 함수 (COT 방식)
def generate_response(user_input):
    """
    사용자의 입력을 받아 FAISS 검색 후 응답 생성 (COT 적용)
    """
    # 사용자의 질문을 임베딩으로 변환
    query_embedding = model_embedding.encode([user_input])

    # FAISS 검색 수행
    search_results = search_faiss(query_embedding)

    # 검색된 결과를 텍스트 형식으로 변환
    search_results_str = "\n".join([str(result) for result in search_results])

    # PromptTemplate에 검색된 결과와 대화 기록 채우기
    filled_prompt = prompt_template.format(
        input_text=user_input,
        search_results=search_results_str,
        chat_history=memory.load_memory_variables({})["chat_history"]
    )

    # 1회 호출에서 5000 토큰 제한이므로 적절하게 텍스트를 나누어 처리
    response_parts = []
    while filled_prompt:
        # 최대 5000 토큰까지 잘라서 호출
        part = filled_prompt[:5000]
        filled_prompt = filled_prompt[5000:]

        # Google Generative AI API 호출 (대신 사용할 모델로 수정 가능)
        response = chat_model.invoke([{"role": "user", "content": part}])
        response_parts.append(response.content)

        # 호출 횟수 체크
        if len(response_parts) >= 3:
            break  # 최대 3회 호출 제한

    # 메모리에 대화 기록 저장
    for part in response_parts:
        memory.save_context({"input": user_input}, {"output": part})

    # 최종 응답 합치기
    return "\n".join(response_parts)

# 챗봇 대화 루프
def chat():
    print("챗봇 대화를 시작합니다. 'exit'을 입력하면 종료됩니다.")
    while True:
        user_input = input("질문을 입력하세요: ")
        if user_input.lower() == "exit":
            break
        try:
            answer = generate_response(user_input)
            print("챗봇 응답:", answer)
        except Exception as e:
            print("오류 발생:", str(e))

# 챗봇 실행
chat()

# Streamlit 페이지 설정
st.set_page_config(page_title="ChatGPT", page_icon="🌴")
st.title("🌴 빅콘테스트 ChatGPT")

# 사용자 입력을 위한 텍스트 입력창
st.write("제주도 맛집 다 알려드림")
message = st.text_input("어떤 맛집을 가고 싶어?", key="input")

# 대화 기록을 저장할 리스트
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 버튼을 눌렀을 때 Ollama 모델 호출
if st.button("전송"):
    if message:
        # 모델 호출 및 응답 받기
        response = model.invoke(message)
        
        # 대화 기록에 추가
        st.session_state['history'].append({"user": message, "bot": response.content})

# 대화 기록을 화면에 출력
if st.session_state['history']:
    for chat in st.session_state['history']:
        st.write(f"**사용자**: {chat['user']}")
        st.write(f"**AI**: {chat['bot']}")


## Chathpt 
import os
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import faiss

# 경로 설정
data_path = '/.data'
module_path = '.modules'

# Gemini 설정
chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# CSV 파일과 .npy 파일 경로 설정
csv_file_paths = [
    'FINAL_REVIEW.csv',
    'FINAL_MCT.csv',
    'FINAL_TRRSRT.csv'
]
npy_file_paths = ['V2/all_embeddings_v2.npy']
index_faiss_path = 'combined_db.index'

# CSV 파일 로드 (low_memory=False로 DtypeWarning 방지)
dfs = [pd.read_csv(csv_file_path, low_memory=False) for csv_file_path in csv_file_paths]

# .npy 파일 로드
embeddings = np.load(npy_file_paths[0])

# FAISS 인덱스 불러오기
faiss_db = faiss.read_index(index_faiss_path)

# FAISS 인덱스에 저장된 총 벡터 개수 
faiss_db.add(embeddings)

# 임베딩 모델 로드 (예: 'jhgan/ko-sroberta-multitask')
model_embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 멀티턴 대화를 위한 Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 멀티턴 프롬프트 템플릿 설정 (COT 방식 적용)
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""(프롬프트 내용은 위와 동일)"""
)

# 4. 검색 및 응답 생성 함수
def search_faiss(query_embedding, k=5):
    """
    FAISS에서 유사한 벡터를 검색하여 원본 데이터 반환
    """
    # FAISS 인덱스에서 유사한 벡터 검색
    distances, indices = faiss_db.search(np.array(query_embedding, dtype=np.float32), k)

    # 검색된 인덱스를 바탕으로 원본 데이터 가져오기
    search_results = []
    for idx in indices[0]:
        for df in dfs:
            if idx < len(df):
                search_results.append(df.iloc[idx])
                break
            else:
                idx -= len(df)  # 인덱스가 초과되면 다음 데이터셋으로 넘어감

    return search_results

# 5. 대화형 응답 생성 함수 (COT 방식)
def generate_response(user_input):
    """
    사용자의 입력을 받아 FAISS 검색 후 응답 생성 (COT 적용)
    """
    # 사용자의 질문을 임베딩으로 변환
    query_embedding = model_embedding.encode([user_input])

    # FAISS 검색 수행
    search_results = search_faiss(query_embedding)

    # 검색된 결과를 텍스트 형식으로 변환
    search_results_str = "\n".join([str(result) for result in search_results])

    # PromptTemplate에 검색된 결과와 대화 기록 채우기
    filled_prompt = prompt_template.format(
        input_text=user_input,
        search_results=search_results_str,
        chat_history=memory.load_memory_variables({})["chat_history"]
    )

    # 1회 호출에서 5000 토큰 제한이므로 적절하게 텍스트를 나누어 처리
    response_parts = []
    while filled_prompt:
        # 최대 5000 토큰까지 잘라서 호출
        part = filled_prompt[:5000]
        filled_prompt = filled_prompt[5000:]

        # Google Generative AI API 호출 (대신 사용할 모델로 수정 가능)
        response = chat_model.invoke([{"role": "user", "content": part}])
        response_parts.append(response.content)

        # 호출 횟수 체크
        if len(response_parts) >= 3:
            break  # 최대 3회 호출 제한

    # 메모리에 대화 기록 저장
    for part in response_parts:
        memory.save_context({"input": user_input}, {"output": part})

    # 최종 응답 합치기
    return "\n".join(response_parts)

# Streamlit 페이지 설정
st.set_page_config(page_title="ChatGPT", page_icon="🌴")
st.title("🌴 제주도 맛집 및 관광지 추천 챗봇")

# 사용자 입력을 위한 텍스트 입력창
st.write("제주도 맛집과 관광지를 추천받아보세요!")
message = st.text_input("어떤 맛집이나 관광지를 찾고 싶으신가요?", key="input")

# 대화 기록을 저장할 리스트
if 'history' not in st.session_state:
    st.session_state['history'] = []

# 버튼을 눌렀을 때 모델 호출
if st.button("질문하기"):
    if message:
        # 모델 호출 및 응답 받기
        response = generate_response(message)  # 모델 호출을 generate_response로 수정
        
        # 대화 기록에 추가
        st.session_state['history'].append({"user": message, "bot": response})

# 대화 기록을 화면에 출력
if st.session_state['history']:
    for chat in st.session_state['history']:
        st.write(f"**사용자**: {chat['user']}")
        st.write(f"**챗봇**: {chat['bot']}")
