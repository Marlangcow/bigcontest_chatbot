import streamlit as st
import pandas as pd
import numpy as np
from recommend import (
    recommend_by_store,
    recommend_by_keyword,
    recommend_popular,
    recommend_by_user_preference,
    df_unique,
)

# -------------------------------
# 📌 Streamlit UI 설정
# -------------------------------
st.title("🏝️ 제주도 맛집 추천 시스템 🏝️")
st.sidebar.header("🔍 필터링 옵션")

# ✅ 추천 방식 선택 (사이드바)
recommendation_type = st.sidebar.radio(
    "추천 방식 선택", ["키워드 검색", "맞춤 추천", "인기 맛집", "유사한 맛집 찾기"]
)

# ✅ 필터 옵션 설정
search_keyword = st.sidebar.text_input("🔍 키워드 검색")

regions = ["전체"] + sorted(df_unique["REGION"].unique().tolist())
selected_region = st.sidebar.selectbox("지역 선택", regions)

categories = ["전체"] + sorted(df_unique["MCT_TYPE"].unique().tolist())
selected_category = st.sidebar.selectbox("업종 선택", categories)

seasons = ["전체", "봄", "여름", "가을", "겨울"]
selected_season = st.sidebar.selectbox("계절 선택", seasons)

weekend_options = ["전체", "주중 선호", "주말 선호", "균형적"]
selected_weekend = st.sidebar.selectbox("주중/주말 선호", weekend_options)

# ✅ 요일 및 시간대 필터
days = ["전체", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
day_columns = [
    "MON_UE_CNT_RAT",
    "TUE_UE_CNT_RAT",
    "WED_UE_CNT_RAT",
    "THU_UE_CNT_RAT",
    "FRI_UE_CNT_RAT",
    "SAT_UE_CNT_RAT",
    "SUN_UE_CNT_RAT",
]
selected_day = st.sidebar.selectbox("방문 요일", days)

times = [
    "전체",
    "오전(5-11시)",
    "점심(12-13시)",
    "오후(14-17시)",
    "저녁(18-22시)",
    "심야(23-4시)",
]
time_columns = [
    "HR_5_11_UE_CNT_RAT",
    "HR_12_13_UE_CNT_RAT",
    "HR_14_17_UE_CNT_RAT",
    "HR_18_22_UE_CNT_RAT",
    "HR_23_4_UE_CNT_RAT",
]
selected_time = st.sidebar.selectbox("방문 시간대", times)

gender_options = ["전체", "남성", "여성"]
selected_gender = st.sidebar.selectbox("고객 성별", gender_options)

age_groups = ["전체", "20세 미만", "30대", "40대", "50대", "60세 이상"]
selected_age_group = st.sidebar.selectbox("고객 연령대", age_groups)

# -------------------------------
# 📌 추천 시스템 기능 구현
# -------------------------------


def filter_data(df):
    """선택된 필터링 옵션에 맞춰 데이터를 필터링"""
    filtered_data = df.copy()

    if selected_region != "전체":
        filtered_data = filtered_data[filtered_data["REGION"] == selected_region]

    if selected_category != "전체":
        filtered_data = filtered_data[filtered_data["MCT_TYPE"] == selected_category]

    if selected_weekend != "전체":
        if selected_weekend == "주중 선호":
            filtered_data = filtered_data[filtered_data["WEEKEND_PREFERENCE"] < 0.9]
        elif selected_weekend == "주말 선호":
            filtered_data = filtered_data[filtered_data["WEEKEND_PREFERENCE"] > 1.1]
        else:
            filtered_data = filtered_data[
                (filtered_data["WEEKEND_PREFERENCE"] >= 0.9)
                & (filtered_data["WEEKEND_PREFERENCE"] <= 1.1)
            ]

    if selected_season != "전체":
        filtered_data = filtered_data[filtered_data["PEAK_SEASON"] == selected_season]

    if selected_gender != "전체":
        filtered_data = filtered_data[filtered_data["MAIN_GENDER"] == selected_gender]

    if selected_age_group != "전체":
        filtered_data = filtered_data[
            filtered_data["MAIN_AGE_GROUP"] == selected_age_group
        ]

    if selected_day != "전체":
        filtered_data = filtered_data.sort_values(
            day_columns[days.index(selected_day) - 1], ascending=False
        )

    if selected_time != "전체":
        filtered_data = filtered_data.sort_values(
            time_columns[times.index(selected_time) - 1], ascending=False
        )

    return filtered_data.sort_values("popularity_score", ascending=False).head(5)


# ✅ 추천 방식 적용
if recommendation_type == "인기 맛집":
    top_places = filter_data(df_unique)
    st.subheader("🔥 인기 맛집 TOP 5")
    for i, row in top_places.iterrows():
        with st.expander(f"{i+1}. {row['MCT_NM']} ({row['MCT_TYPE']})"):
            st.write(f"**📍 주소:** {row['ADDR']}")
            st.write(f"**⭐ 평점:** {row['popularity_score']:.2f}")

elif recommendation_type == "유사한 맛집 찾기":
    store_names = df_unique["MCT_NM"].unique().tolist()
    selected_store = st.sidebar.selectbox("찾고 싶은 가게를 선택하세요", store_names)

    if selected_store:
        similar_places = recommend_by_store(selected_store)

        if isinstance(similar_places, str):  # 유사한 맛집이 없을 경우
            st.warning(similar_places)
        else:
            st.subheader(f"🔍 '{selected_store}'와 유사한 맛집")
            for i, row in similar_places.iterrows():
                with st.expander(f"{i+1}. {row['MCT_NM']} ({row['MCT_TYPE']})"):
                    st.write(f"**📍 주소:** {row['ADDR']}")
                    st.write(f"**⭐ 평점:** {row['popularity_score']:.2f}")

elif recommendation_type == "맞춤 추천":
    top_places = filter_data(df_unique)
    st.subheader("🎯 맞춤 추천 맛집")
    for i, row in top_places.iterrows():
        with st.expander(f"{i+1}. {row['MCT_NM']} ({row['MCT_TYPE']})"):
            st.write(f"**📍 주소:** {row['ADDR']}")
            st.write(f"**⭐ 평점:** {row['popularity_score']:.2f}")

elif recommendation_type == "키워드 검색":
    if search_keyword:
        recommended_places = recommend_by_keyword(search_keyword)

        if isinstance(recommended_places, str):
            st.warning(recommended_places)
        else:
            st.subheader(f"🔎 '{search_keyword}' 관련 추천 맛집")
            for i, row in recommended_places.iterrows():
                with st.expander(f"{i+1}. {row['MCT_NM']} ({row['MCT_TYPE']})"):
                    st.write(f"**📍 주소:** {row['ADDR']}")
                    st.write(f"**⭐ 평점:** {row['popularity_score']:.2f}")
