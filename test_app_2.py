import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import faiss

# Streamlit 페이지 설정
st.set_page_config(page_title="🍊감귤톡")

# Streamlit App UI
st.title("🍊감귤톡, 제주도 여행 메이트")
st.info("제주도 여행 메이트 감귤톡이 제주도의 방방곡곡을 알려줄게🌴")

# 이미지 로드 설정
if 'image_loaded' not in st.session_state:
    st.session_state.image_html = """
    <div style="display: flex; justify-content: center;">
        <img src="https://img4.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202105/25/linkagelab/20210525013157546odxh.jpg" alt="centered image" width="50%">
    </div>
    """
    st.session_state.image_loaded = True

st.write("")  # 여백 추가

# .env 파일 경로 지정
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# CSV 파일 로드
@st.cache_data

# CSV 파일 로드
def load_data():
    csv_file_paths = [
        './data/review_documents.csv',
        './data/mct_documents.csv',
        './data/trrsrt_documents.csv'
    ]
    dfs = []
    
    with st.spinner("잠시만 기다려 주세요. 곧 나와요!"):  # 사용자 정의 스피너 메시지
        dfs = [pd.read_csv(csv_file_path) for csv_file_path in csv_file_paths]
    
    return dfs

dfs = load_data()

# LLM에서 사용할 함수 tool 생성
def create_tool(name, description, required_params):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {param: {"type": "string", "description": f"{param}에 대한 설명"} for param in required_params},
                "required": required_params
            }
        }
    }

# 각각의 툴 생성
review_tool = create_tool("get_review_data", "맛집과 관광지 리뷰 정보를 가져옵니다.", ["장소"])
mct_tool = create_tool("get_mct_data", "MCT 문서의 정보를 조회합니다.", ["가게명"])
trrsrt_tool = create_tool("get_trrsrt_data", "관광지 문서의 정보를 조회합니다.", ["관광지명"])

tools = [review_tool, mct_tool, trrsrt_tool]


# FAISS 인덱스 및 임베딩 모델 로드
faiss_index_path = './modules/faiss_index.index'
faiss_index = faiss.read_index(faiss_index_path)


# 임베딩 모델 로드
def load_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

model_embedding = load_model()


# LLM 설정
chat_model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key=google_api_key,
    temperature=0.3,
    top_p=0.85,
    frequency_penalty=0.3
)


# 멀티턴 대화를 위한 Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 멀티턴 프롬프트 템플릿 설정 (COT 방식 적용)
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""
   ### 역할
    당신은 제주도 맛집과 관광지 추천 전문가입니다. 질문을 받을 때 논리적으로 생각한 후 단계별로 답변을 제공합니다.
    복잡한 질문일수록 천천히 생각하고 검색된 데이터를 바탕으로 답변을 제공합니다.

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
    1. 검색할 내용이 충분하지 않다면 사용자에게 반문하세요. 이는 가장 중요합니다. 단, 두번 이상 반문하지 마세요. 만약 사용자가 위치를 모른다면 제일 평점이 좋은 3개의 식당+카페와 3개의 관광지를 안내해주세요.
    2. 친근하고 재미있으면서도 정겹게 안내하세요.
    3. source_id는 문서 번호입니다. 따라서 답변을 하는 경우 몇 번 문서를 인용했는지 답변 뒤에 '(문서 번호: source_id1)' 형식으로 보여주세요.
    4. 추천 할 때, 추천 이유와 소요되는 거리, 평점과 리뷰들도 보여줘. 만약 리뷰가 없는 곳이라면 ("작성된 리뷰가 없습니다.") 라고 해주세요.
    5. 4번의 지시사항과 함께 판매 메뉴 2개, 가격도 알려주세요.
    6. 위도와 경도를 바탕으로 실제 검색되는 장소를 https://map.naver.com/p/search/제주도 <placename>장소이름</placename>으로 답변하세요. 단, 지도 링크가 없는 곳은 지도 링크라는 문구를 아예 노출하지 말아주세요.
        예시 링크는 다음과 같습니다. <placename> 태그는 알기 쉽게 구분 해 놓은 값이며, 절대 링크내에 삽입되어서는 안됩니다.
        - https://map.naver.com/p/search/제주도+우진해장국/
        - https://map.naver.com/p/search/제주도+카페봄날/
        - https://map.naver.com/p/search/제주도+고흐의정원/
    7. 실제로 존재하는 식당과 관광지명을 추천해주어야 하며, %%흑돼지 맛집, 횟집 1 등 가게명이 명확하지 않은 답변은 하지 말아주세요.
    8. 문장이 구분되도록 문단을 구분해주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """
)

# 검색 및 응답 생성 함수
def search_faiss(query_embedding, k=5):
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)
    search_results = []

    for idx in indices[0]:
        for df in dfs:
            if idx < len(df):
                search_results.append(df.iloc[idx])
                break
            idx -= len(df)

    return search_results


# 대화형 응답 생성 함수
def generate_response(user_input):
    query_embedding = model_embedding.encode([user_input])
    search_results = search_faiss(query_embedding)

    search_results_str = "\n".join([result.to_string() for result in search_results])
    
    filled_prompt = prompt_template(
        input_text=user_input,
        search_results=search_results_str,
        chat_history=memory.load_memory_variables({})["chat_history"]
    )

    response = chat_model.invoke([{"role": "user", "content": filled_prompt}], tools=tools)

    # 응답 처리
    result = response.content
    if response["finish_reason"] == "function_call":
        function_data = response["tool_call"]["arguments"]
        if "리뷰" in function_data:
            result = get_review_data(**function_data)
        elif "가게명" in function_data:
            result = get_mct_data(**function_data)
        elif "관광지명" in function_data:
            result = get_trrsrt_data(**function_data)

    memory.save_context({"input": user_input}, {"output": result})
    return result

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


# 스트림릿 챗봇 인터페이스
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 이미지 표시 (세션 상태 유지)
st.markdown(st.session_state.image_html, unsafe_allow_html=True)

# 이전 대화 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)

