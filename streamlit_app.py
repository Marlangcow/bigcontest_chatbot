import streamlit as st
from langchain_community.chat_models import ChatOllama

# Streamlit 페이지 설정
st.set_page_config(page_title="ChatGPT", page_icon="🌴")
st.title("🌴 빅콘테스트 ChatGPT")

# Ollama 모델 로드
model = ChatOllama(model="llama3:8b", temperature=0)

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
        st.write(f"**Ollama**: {chat['bot']}")
