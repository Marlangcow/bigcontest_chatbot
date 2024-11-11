import streamlit as st


def initialize_streamlit_ui():
    st.set_page_config(
        page_title="🍊감귤톡",
        page_icon="🍊",
        layout="wide",
    )

    st.title("🍊감귤톡, 제주도 여행 메이트")
    st.write("")
    st.info("제주도 여행 메이트 감귤톡이 제주도의 방방곡곡을 알려줄게 🏝️")

    # 이미지 표시
    display_main_image()

    with st.sidebar:
        setup_sidebar()


def display_main_image():
    image_path = "https://img4.daumcdn.net/thumb/R658x0.q70/?fname=https://t1.daumcdn.net/news/202105/25/linkagelab/20210525013157546odxh.jpg"
    image_html = f"""
    <div style="display: flex; justify-content: center;">
        <img src="{image_path}" alt="centered image" width="50%">
    </div>
    """
    st.markdown(image_html, unsafe_allow_html=True)
    st.write("")


def setup_sidebar():
    st.title("🍊감귤톡이 다 찾아줄게🍊")
    st.write("")
    setup_keyword_selection()
    setup_location_selection()
    setup_score_selection()
    st.button("대화 초기화", on_click=clear_chat_history)
    st.caption("📨 감귤톡에 문의하기 [Send email](mailto:happily2bus@gmail.com)")


def setup_keyword_selection():
    st.subheader("원하는 #키워드를 골라봐")
    remove_selectbox_label()
    keywords = st.selectbox(
        "",
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
        key="keywords",
    )
    st.write("")


def setup_location_selection():
    st.subheader("어떤 장소가 궁금해?")
    remove_radio_label()
    locations = st.selectbox(
        "",
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
    )
    st.write("")


def setup_score_selection():
    st.subheader("평점 몇점 이상을 원해?")
    score = st.slider("리뷰 평점", min_value=3.0, max_value=5.0, value=4.5, step=0.05)
    st.write("")


def remove_selectbox_label():
    st.markdown(
        """
        <style>
        .stSelectbox label { display: none; }
        .stSelectbox div[role='combobox'] { margin-top: -20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def remove_radio_label():
    st.markdown(
        """
        <style>
        .stRadio > label { display: none; }
        .stRadio > div { margin-top: -20px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(
            message["role"], avatar="🐬" if message["role"] == "assistant" else None
        ):
            st.markdown(message["content"])


def handle_streamlit_input(chain, memory):
    if prompt := st.chat_input("질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = get_chatbot_response(prompt, memory, chain)

        with st.chat_message("assistant", avatar="🐬"):
            with st.spinner("생각하는 중..."):
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어떤 곳을 찾아줄까?"}
    ]
