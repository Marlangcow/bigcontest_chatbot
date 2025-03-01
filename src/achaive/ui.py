import streamlit as st
from src.chatbot import ChatBot
from src.data_loader import initialize_retrievers

# st.session_state.memory 초기화
if "memory" not in st.session_state:
    st.session_state.memory = None  # 초기값 설정


def initialize_streamlit_ui():
    # st.session_state.messages 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
        ]

    # 제목 및 정보 텍스트 설정
    st.title("🍊감귤톡, 제주도 여행 메이트")
    st.write("")
    st.info("제주도 여행 메이트 감귤톡이 제주도의 방방곡곡을 알려줄게 🏝️")

    # 이미지 표시
    display_main_image()

    # 사이드바 설정
    setup_sidebar()


def display_main_image():
    image_path = "https://img4.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202105/25/linkagelab/20210525013157546odxh.jpg"
    st.image(image_path, use_container_width=False)
    st.write("")


def setup_sidebar():
    st.sidebar.title("🍊감귤톡이 다 찾아줄게🍊")
    st.sidebar.write("")
    setup_common_ui_elements()


def setup_common_ui_elements():
    """사이드바와 메인 UI에서 공통으로 사용되는 요소 설정"""
    initialize_chat()
    setup_keyword_selection()
    setup_location_selection()
    setup_score_selection()
    st.sidebar.button("대화 초기화", on_click=clear_chat_history)
    st.sidebar.write("")
    st.sidebar.caption("📨 감귤톡 제작자: 8Lee8Lee, Marlangcow")


def setup_keyword_selection():
    st.subheader("원하는 #키워드를 골라봐")
    keywords = st.selectbox(
        "키워드 선택",
        [
            "착한가격업소",
            "럭셔리트래블인제주",
            "우수관광사업체",
            "무장애관광",
            "안전여행스탬프",
            "향토음식",
            "한식",
            "카페",
            "해물뚝배기",
            "몸국",
            "해장국",
            "수제버거",
            "흑돼지",
            "해산물",
            "일식",
        ],
        key="visit_keywords",
        label_visibility="collapsed",
    )
    st.write("")


def setup_location_selection():
    st.subheader("어떤 장소가 궁금해?")
    locations = st.selectbox(
        "장소 선택",
        [
            "구좌",
            "대정",
            "서귀포",
            "안덕",
            "우도",
            "애월",
            "조천",
            "제주시내",
            "추자",
            "한림",
            "한경",
        ],
        key="visit_locations",
        label_visibility="collapsed",
    )
    st.write("")


def setup_score_selection():
    st.subheader("평점 몇점 이상을 원해?")
    score = st.slider("리뷰 평점", min_value=3.0, max_value=5.0, value=4.5, step=0.05)


st.write("")


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]


def initialize_chat():
    if "chatbot" not in st.session_state:
        retrievers = initialize_retrievers()
        st.session_state.chatbot = ChatBot(retrievers)
