# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import folium
from streamlit_folium import folium_static
import random
from PIL import Image
import io

# 프로젝트 모듈 임포트
import config
from src.data_processor import build_features_dataset
from src.recommender import recommend_jeju_places, train_collaborative_model

# 페이지 설정
st.set_page_config(
    page_title="제주 관광지 추천 시스템",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 스타일 설정
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .recommendation-card {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .place-name {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1565C0;
    }
    .place-info {
        margin-top: 0.5rem;
        color: #424242;
    }
    .score-badge {
        background-color: #E3F2FD;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-weight: bold;
        color: #0D47A1;
    }
    .keyword-tag {
        display: inline-block;
        background-color: #E1F5FE;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.8rem;
        color: #01579B;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #757575;
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 세션 상태 초기화
if "user_id" not in st.session_state:
    st.session_state.user_id = random.randint(1000, 9999)
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "show_map" not in st.session_state:
    st.session_state.show_map = False


# 데이터 및 모델 로드 함수
@st.cache_resource
def load_data_and_model():
    """데이터 및 모델 로드 (캐싱)"""
    # 특성 데이터 로드 또는 구축
    if os.path.exists(config.FEATURES_PATH):
        features = pd.read_pickle(config.FEATURES_PATH)
    else:
        features = build_features_dataset()

    # 평점 데이터 로드
    if os.path.exists(config.RATINGS_PATH):
        ratings_data = pd.read_pickle(config.RATINGS_PATH)
    else:
        st.error("평점 데이터를 찾을 수 없습니다. 데이터 처리를 먼저 실행해주세요.")
        return None, None

    # 모델 로드 또는 학습
    if os.path.exists(config.MODEL_PATH):
        with open(config.MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        model = train_collaborative_model(ratings_data)

    return features, model


# 워드클라우드 생성 함수
def generate_wordcloud(text):
    """키워드로 워드클라우드 생성"""
    if not text or pd.isna(text):
        return None

    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color="white",
        max_words=50,
        font_path="malgun",  # 한글 폰트 경로 (필요시 수정)
        colormap="viridis",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


# 지도 생성 함수
def create_map(recommendations):
    """추천 장소 지도 시각화"""
    # 제주도 중심 좌표
    jeju_center = [33.3846, 126.5535]
    m = folium.Map(location=jeju_center, zoom_start=10)

    # 주소를 좌표로 변환하는 함수 (실제로는 지오코딩 API 사용 필요)
    # 여기서는 간단한 예시로 랜덤 좌표 생성
    def get_coordinates(address):
        # 제주시/서귀포시 구분하여 대략적인 좌표 생성
        if "제주시" in address:
            lat = 33.5 + random.uniform(-0.1, 0.1)
            lon = 126.5 + random.uniform(-0.2, 0.2)
        else:
            lat = 33.25 + random.uniform(-0.1, 0.1)
            lon = 126.5 + random.uniform(-0.2, 0.2)
        return lat, lon

    # 각 추천 장소에 마커 추가
    for i, row in recommendations.iterrows():
        place_name = row["AREA_NM"]
        address = row["ADDR"]
        rating = row.get("rating", 0)
        score = row.get("total_score", 0)

        # 좌표 얻기
        lat, lon = get_coordinates(address)

        # 마커 색상 (점수에 따라)
        color = "red" if score > 0.8 else "orange" if score > 0.6 else "blue"

        # 팝업 내용
        popup_html = f"""
        <div style="width:200px">
            <h4>{place_name}</h4>
            <p><b>주소:</b> {address}</p>
            <p><b>평점:</b> {rating:.1f}/5.0</p>
            <p><b>추천 점수:</b> {score:.2f}</p>
        </div>
        """

        # 마커 추가
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=place_name,
            icon=folium.Icon(color=color),
        ).add_to(m)

    return m


# 메인 앱 함수
def main():
    # 헤더
    st.markdown(
        '<h1 class="main-header">🏝️ 제주 관광지 추천 시스템</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center">당신의 취향에 맞는 제주도 관광지를 추천해드립니다</p>',
        unsafe_allow_html=True,
    )

    # 사이드바
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🔍 추천 설정</h2>', unsafe_allow_html=True)

        # 데이터 로드 버튼
        if st.button("데이터 및 모델 로드"):
            with st.spinner("데이터와 모델을 로드하는 중..."):
                features, model = load_data_and_model()
                if features is not None and model is not None:
                    st.session_state.data_loaded = True
                    st.session_state.model_loaded = True
                    st.success("데이터와 모델 로드 완료!")

        # 데이터가 로드된 경우에만 추천 설정 표시
        if st.session_state.data_loaded:
            st.markdown("### 👤 사용자 정보")
            st.write(f"사용자 ID: {st.session_state.user_id}")

            gender = st.radio("성별", ["여성", "남성"])
            is_tourist = st.checkbox("관광객", value=True)

            st.markdown("### 🗺️ 지역 설정")
            region = st.selectbox("지역", ["제주시", "서귀포시", "전체"])

            st.markdown("### 🏨 업종 설정")
            business_type = st.selectbox(
                "업종", ["전체", "관광지", "숙박", "음식점", "카페"]
            )
            if business_type == "전체":
                business_type = None

            st.markdown("### ⭐ 필터링 설정")
            min_rating = st.slider("최소 평점", 0.0, 5.0, 3.5, 0.1)

            st.markdown("### 🔑 키워드 설정")
            keyword_input = st.text_input("키워드 (쉼표로 구분)", "뷰,분위기")
            keywords = (
                [k.strip() for k in keyword_input.split(",")] if keyword_input else []
            )

            st.markdown("### 📅 방문 패턴")
            weekend_preference = st.checkbox("주말 선호", value=True)

            st.markdown("### 🌞 계절 선호도")
            season_preference = st.selectbox(
                "선호 계절", ["여름", "겨울", "봄", "가을", "상관없음"]
            )
            if season_preference == "상관없음":
                season_preference = None

            # 추천 버튼
            if st.button("추천 받기"):
                with st.spinner("추천을 생성하는 중..."):
                    recommendations = recommend_jeju_places(
                        user_id=st.session_state.user_id,
                        gender=gender,
                        is_tourist=is_tourist,
                        region=region if region != "전체" else None,
                        min_rating=min_rating,
                        keywords=keywords,
                        weekend_preference=weekend_preference,
                        season_preference=season_preference,
                        business_type=business_type,
                    )

                    if recommendations is not None and not recommendations.empty:
                        st.session_state.recommendations = recommendations
                        st.success("추천이 완료되었습니다!")
                    else:
                        st.error(
                            "추천을 생성할 수 없습니다. 필터링 조건을 완화해보세요."
                        )

        else:
            st.info("먼저 '데이터 및 모델 로드' 버튼을 클릭하여 데이터를 로드해주세요.")

    # 메인 컨텐츠
    if not st.session_state.data_loaded:
        # 데이터가 로드되지 않은 경우 안내 메시지
        st.info("사이드바에서 '데이터 및 모델 로드' 버튼을 클릭하여 시작해주세요.")

        # 제주도 이미지 표시
        st.image(
            "https://www.visitjeju.net/ckImage/202110/ckeditor_3458570986499832474.jpg",
            caption="제주도의 아름다운 풍경",
            use_column_width=True,
        )

        # 시스템 소개
        st.markdown(
            """
        ## 🌊 제주 관광지 추천 시스템 소개
        
        이 시스템은 제주도의 다양한 관광지, 숙박, 음식점, 카페 등을 사용자의 취향에 맞게 추천해드립니다.
        
        ### 주요 기능:
        - 🔍 **개인화된 추천**: 협업 필터링을 통한 맞춤형 추천
        - 🏷️ **키워드 기반 검색**: 원하는 키워드가 포함된 장소 추천
        - 📊 **데이터 기반 분석**: 방문 패턴, 계절별 인기도 등 다양한 데이터 활용
        - 🗺️ **지도 시각화**: 추천 장소를 지도에서 확인
        
        ### 사용 방법:
        1. 사이드바에서 '데이터 및 모델 로드' 버튼 클릭
        2. 사용자 정보와 선호도 설정
        3. '추천 받기' 버튼 클릭
        4. 추천 결과 확인
        """
        )

    elif st.session_state.recommendations is not None:
        # 추천 결과가 있는 경우 표시
        recommendations = st.session_state.recommendations

        # 탭 생성
        tab1, tab2, tab3 = st.tabs(["📋 추천 목록", "📊 데이터 분석", "🗺️ 지도 보기"])

        with tab1:
            st.markdown(
                '<h2 class="sub-header">📋 추천 장소 목록</h2>', unsafe_allow_html=True
            )

            # 각 추천 장소 카드 형태로 표시
            for i, row in recommendations.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                    <div class="recommendation-card">
                        <div class="place-name">{i+1}. {row['AREA_NM']}</div>
                        <div class="place-info">
                            <p><b>업종:</b> {row['CL_NM']}</p>
                            <p><b>주소:</b> {row['ADDR']}</p>
                            <p>
                                <span class="score-badge">평점: {row.get('rating', 0):.1f}/5.0</span>
                                <span class="score-badge">리뷰 수: {int(row.get('review_count', 0))}</span>
                                <span class="score-badge">추천 점수: {row.get('total_score', 0):.2f}</span>
                            </p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # 키워드 표시 (있는 경우)
                    if "keywords" in row and not pd.isna(row["keywords"]):
                        keywords = row["keywords"].split()[:10]  # 상위 10개만 표시
                        keyword_html = '<div style="margin-top:0.5rem;">'
                        for kw in keywords:
                            keyword_html += f'<span class="keyword-tag">{kw}</span>'
                        keyword_html += "</div>"
                        st.markdown(keyword_html, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown(
                '<h2 class="sub-header">📊 추천 데이터 분석</h2>',
                unsafe_allow_html=True,
            )

            # 평점 분포
            st.subheader("평점 분포")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(recommendations["rating"], bins=10, kde=True, ax=ax)
            ax.set_xlabel("평점")
            ax.set_ylabel("빈도")
            st.pyplot(fig)

            # 업종별 분포
            st.subheader("업종별 분포")
            fig, ax = plt.subplots(figsize=(10, 5))
            recommendations["CL_NM"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel("업종")
            ax.set_ylabel("개수")
            st.pyplot(fig)

            # 키워드 워드클라우드 (모든 추천의 키워드 합침)
            st.subheader("키워드 워드클라우드")
            if "keywords" in recommendations.columns:
                all_keywords = " ".join(recommendations["keywords"].dropna())
                if all_keywords:
                    wordcloud_fig = generate_wordcloud(all_keywords)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                else:
                    st.info("키워드 데이터가 없습니다.")
            else:
                st.info("키워드 데이터가 없습니다.")

        with tab3:
            st.markdown(
                '<h2 class="sub-header">🗺️ 추천 장소 지도</h2>', unsafe_allow_html=True
            )
            st.info("지도에 표시된 위치는 실제 위치와 다를 수 있습니다. (예시용)")

            # 지도 생성
            map_obj = create_map(recommendations)
            folium_static(map_obj, width=800, height=600)

    # 푸터
    st.markdown(
        """
    <div class="footer">
        <p>© 2023 제주 관광지 추천 시스템 | 데이터 출처: 제주 관광 데이터</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
