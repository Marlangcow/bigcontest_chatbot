import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import time
import matplotlib.pyplot as plt


# 데이터 로드 함수 수정
@st.cache_resource
def load_recommendation_data():
    try:
        # CSV 파일 로드
        merged_data = pd.read_csv("notebooks/processed_tourism_data.csv")

        # TF-IDF 벡터화
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data["features"])

        # 유사도 행렬 계산
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return {
            "merged_data": merged_data,
            "tfidf_vectorizer": tfidf_vectorizer,
            "similarity_matrix": similarity_matrix,
        }
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
        return None


# 데이터 로드
try:
    recommendation_data = load_recommendation_data()
    if recommendation_data:
        merged_data = recommendation_data["merged_data"]
        tfidf_vectorizer = recommendation_data["tfidf_vectorizer"]
        similarity_matrix = recommendation_data["similarity_matrix"]
        st.success("추천 시스템 데이터가 성공적으로 로드되었습니다!")
    else:
        st.error("추천 시스템 데이터를 로드할 수 없습니다.")
        st.stop()
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()


# 앱 제목
st.title("🏝️제주도 관광지 추천 시스템🏝️")

# 사이드바 - 필터링 옵션
st.sidebar.header("필터링 옵션")

# 지역 선택 (지역 컬럼이 있는 경우에만)
if "지역" in merged_data.columns:
    # 모든 값을 문자열로 변환하여 정렬
    region_values = merged_data["지역"].astype(str).unique().tolist()
    regions = ["전체"] + sorted(region_values)
    selected_region = st.sidebar.selectbox("지역 선택", regions, key="sidebar_region")
else:
    selected_region = "전체"

# 카테고리 선택
if "CL_NM" in merged_data.columns:
    # 모든 값을 문자열로 변환하여 정렬
    category_values = merged_data["CL_NM"].astype(str).unique().tolist()
    categories = ["전체"] + sorted(category_values)
    selected_category = st.sidebar.selectbox(
        "카테고리 선택", categories, key="sidebar_category"
    )
else:
    selected_category = "전체"

# 방문 시기 선택
months = [
    "전체",
    "1월",
    "2월",
    "3월",
    "4월",
    "5월",
    "6월",
    "7월",
    "8월",
    "9월",
    "10월",
    "11월",
    "12월",
]
selected_month = st.sidebar.selectbox("방문 시기", months, key="sidebar_month")
month_idx = months.index(selected_month) if selected_month != "전체" else None

# 요일 선택
days = ["전체", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
day_columns = [
    "DAY_1_RATIO",
    "DAY_2_RATIO",
    "DAY_3_RATIO",
    "DAY_4_RATIO",
    "DAY_5_RATIO",
    "DAY_6_RATIO",
    "DAY_7_RATIO",
]
selected_day = st.sidebar.selectbox("방문 요일", days, key="sidebar_day")
day_idx = days.index(selected_day) - 1 if selected_day != "전체" else None

# 시간대 선택 (있는 경우)
if "HR_5_11_UE_CNT_RAT" in merged_data.columns:
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
    selected_time = st.sidebar.selectbox("방문 시간대", times, key="sidebar_time")
    time_idx = times.index(selected_time) - 1 if selected_time != "전체" else None
else:
    selected_time = "전체"
    time_idx = None

# 주중/주말 선택
weekend_options = ["전체", "주중 선호", "주말 선호", "균형적"]
selected_weekend = st.sidebar.selectbox(
    "주중/주말 선호", weekend_options, key="sidebar_weekend"
)

# 계절 선택
season_options = ["전체", "봄", "여름", "가을", "겨울"]
season_mapping = {"봄": "SPRING", "여름": "SUMMER", "가을": "FALL", "겨울": "WINTER"}
selected_season = st.sidebar.selectbox(
    "계절 선호", season_options, key="sidebar_season"
)

# 키워드 검색
search_keyword = st.sidebar.text_input("키워드 검색", key="sidebar_keyword")

# 메인 화면 - 추천 시스템
st.header("관광지 추천")

# 추천 방식 선택
recommendation_type = st.radio(
    "추천 방식 선택",
    ["인기 관광지", "유사 관광지 찾기", "맞춤 추천"],
    key="recommendation_type",
)

# 추천 타입별 로직
if recommendation_type == "인기 관광지":
    # 필터링
    filtered_data = merged_data.copy()

    # 지역 필터링
    if selected_region != "전체" and "지역" in merged_data.columns:
        # 문자열 비교를 위해 컬럼을 문자열로 변환
        filtered_data = filtered_data[
            filtered_data["지역"].astype(str) == selected_region
        ]

    # 카테고리 필터링
    if selected_category != "전체" and "CL_NM" in merged_data.columns:
        # 문자열 비교를 위해 컬럼을 문자열로 변환
        filtered_data = filtered_data[
            filtered_data["CL_NM"].astype(str) == selected_category
        ]

    # 월별 데이터 처리
    if month_idx is not None:
        month_ratio_col = f"MONTH_{month_idx}_RATIO"
        if month_ratio_col in filtered_data.columns:
            filtered_data = filtered_data.sort_values(month_ratio_col, ascending=False)
        else:
            st.warning(
                f"{selected_month} 방문 데이터가 없습니다. 전체 인기도 기준으로 정렬합니다."
            )
            filtered_data = filtered_data.sort_values(
                "popularity_score", ascending=False
            )

    # 요일 데이터 처리
    if day_idx is not None:
        if day_columns[day_idx] in filtered_data.columns:
            filtered_data = filtered_data.sort_values(
                day_columns[day_idx], ascending=False
            )
        else:
            st.warning(
                f"{selected_day} 방문 데이터가 없습니다. 전체 인기도 기준으로 정렬합니다."
            )
            filtered_data = filtered_data.sort_values(
                "popularity_score", ascending=False
            )

    # 시간대 데이터 처리 (있는 경우)
    if time_idx is not None and "HR_5_11_UE_CNT_RAT" in filtered_data.columns:
        if time_columns[time_idx] in filtered_data.columns:
            filtered_data = filtered_data.sort_values(
                time_columns[time_idx], ascending=False
            )
        else:
            st.warning(
                f"{selected_time} 방문 데이터가 없습니다. 전체 인기도 기준으로 정렬합니다."
            )
            filtered_data = filtered_data.sort_values(
                "popularity_score", ascending=False
            )

    # 주중/주말 선호도 필터링
    if selected_weekend != "전체" and "WEEKEND_PREFERENCE" in filtered_data.columns:
        if selected_weekend == "주중 선호":
            filtered_data = filtered_data[filtered_data["WEEKEND_PREFERENCE"] < 0.9]
        elif selected_weekend == "주말 선호":
            filtered_data = filtered_data[filtered_data["WEEKEND_PREFERENCE"] > 1.1]
        elif selected_weekend == "균형적":
            filtered_data = filtered_data[
                (filtered_data["WEEKEND_PREFERENCE"] >= 0.9)
                & (filtered_data["WEEKEND_PREFERENCE"] <= 1.1)
            ]

    # 계절 선호도 필터링
    if selected_season != "전체" and "PEAK_SEASON" in filtered_data.columns:
        season_code = season_mapping.get(selected_season)
        if season_code:
            filtered_data = filtered_data[filtered_data["PEAK_SEASON"] == season_code]

    # 키워드 검색
    if search_keyword:
        filtered_data = filtered_data[
            filtered_data["features"].str.contains(search_keyword, case=False, na=False)
        ]

    # 인기도 순으로 정렬
    top_spots = filtered_data.sort_values("popularity_score", ascending=False).head(5)

    # 결과 표시
    if len(top_spots) > 0:
        st.subheader("인기 관광지 TOP 5")
        for i, (idx, row) in enumerate(top_spots.iterrows()):
            with st.expander(f"{i+1}. {row['AREA_NM']} ({row['CL_NM']})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**지역:** {row['지역']}")
                    st.write(f"**카테고리:** {row['CL_NM']}")
                    st.write(f"**주소:** {row['ADDR']}")
                    if "AVRG_SCORE_VALUE" in row and not pd.isna(
                        row["AVRG_SCORE_VALUE"]
                    ):
                        st.write(f"**평점:** {row['AVRG_SCORE_VALUE']}")
                    if "positive_ratio" in row and not pd.isna(row["positive_ratio"]):
                        st.write(f"**긍정 비율:** {row['positive_ratio']:.2f}")
                    if "PEAK_SEASON" in row and not pd.isna(row["PEAK_SEASON"]):
                        season_korean = {v: k for k, v in season_mapping.items()}.get(
                            row["PEAK_SEASON"], row["PEAK_SEASON"]
                        )
                        st.write(f"**성수기:** {season_korean}")
                    if "WEEKEND_PREFERENCE" in row and not pd.isna(
                        row["WEEKEND_PREFERENCE"]
                    ):
                        if row["WEEKEND_PREFERENCE"] > 1.1:
                            weekend_pref = "주말 선호"
                        elif row["WEEKEND_PREFERENCE"] < 0.9:
                            weekend_pref = "주중 선호"
                        else:
                            weekend_pref = "균형적"
                        st.write(f"**주중/주말:** {weekend_pref}")

                with col2:
                    # 월별 방문 비율 차트
                    try:
                        monthly_columns = [f"MONTH_{i+1}_RATIO" for i in range(12)]
                        existing_monthly_columns = [
                            col for col in monthly_columns if col in row.index
                        ]

                        if existing_monthly_columns:
                            monthly_data = [
                                row[col] for col in existing_monthly_columns
                            ]
                            month_labels = [
                                months[int(col.split("_")[1])]
                                for col in existing_monthly_columns
                            ]
                            st.bar_chart(
                                pd.DataFrame(
                                    {"방문 비율": monthly_data}, index=month_labels
                                )
                            )
                        else:
                            st.info("월별 방문 데이터가 없습니다.")
                    except Exception as e:
                        st.error(f"차트 표시 중 오류 발생: {e}")
                        st.info("월별 방문 데이터를 표시할 수 없습니다.")

                # 키워드 표시
                if (
                    "CORE_KWRD_CN" in row
                    and row["CORE_KWRD_CN"]
                    and not pd.isna(row["CORE_KWRD_CN"])
                ):
                    st.write("**주요 키워드:**")
                    keywords = row["CORE_KWRD_CN"].split(",")
                    st.write(", ".join(keywords))
    else:
        st.info("조건에 맞는 관광지가 없습니다. 필터링 조건을 변경해보세요.")

elif recommendation_type == "유사 관광지 찾기":
    col1, col2 = st.columns([1, 2])
    with col1:
        # 관광지 선택
        spot_names = merged_data["AREA_NM"].unique().tolist()
        selected_spot = st.selectbox("관광지 선택", spot_names, key="similar_spot")

    with col2:
        # 유사도 기준 선택 추가
        similarity_criteria = st.multiselect(
            "유사도 기준 선택",
            ["카테고리", "방문 패턴", "키워드", "성수기"],
            default=["카테고리", "키워드"],
            key="similarity_criteria",
        )

    if selected_spot:
        # 선택한 관광지의 인덱스 찾기
        spot_idx = merged_data[merged_data["AREA_NM"] == selected_spot].index[0]

        # 유사도 가중치 적용
        weights = {"카테고리": 0.3, "방문 패턴": 0.3, "키워드": 0.2, "성수기": 0.2}

        # 선택된 기준에 따라 유사도 계산
        final_similarities = np.zeros(len(merged_data))
        total_weight = 0

        for criterion in similarity_criteria:
            weight = weights[criterion]
            total_weight += weight
            if criterion == "카테고리":
                final_similarities += weight * (
                    merged_data["CL_NM"] == merged_data.loc[spot_idx, "CL_NM"]
                ).astype(float)
            elif criterion == "방문 패턴":
                visit_pattern_similarity = (
                    1
                    - np.abs(
                        merged_data["WEEKEND_PREFERENCE"]
                        - merged_data.loc[spot_idx, "WEEKEND_PREFERENCE"]
                    )
                    / 2
                )
                final_similarities += weight * visit_pattern_similarity
            elif criterion == "키워드":
                final_similarities += weight * similarity_matrix[spot_idx]
            elif criterion == "성수기":
                final_similarities += weight * (
                    merged_data["PEAK_SEASON"]
                    == merged_data.loc[spot_idx, "PEAK_SEASON"]
                ).astype(float)

        # 가중치 정규화
        if total_weight > 0:
            final_similarities /= total_weight

        # 유사한 관광지 인덱스 (자기 자신 제외)
        similar_indices = final_similarities.argsort()[::-1][1:11]
        similar_spots = merged_data.iloc[similar_indices]

        # 결과 표시
        st.subheader(f"{selected_spot}와(과) 유사한 관광지")
        for i, (idx, row) in enumerate(similar_spots.iterrows()):
            similarity_score = final_similarities[idx]
            with st.expander(
                f"{i+1}. {row['AREA_NM']} (유사도: {similarity_score:.2f})"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**지역:** {row['지역']}")
                    st.write(f"**카테고리:** {row['CL_NM']}")
                    st.write(f"**주소:** {row['ADDR']}")
                    if "AVRG_SCORE_VALUE" in row and not pd.isna(
                        row["AVRG_SCORE_VALUE"]
                    ):
                        st.write(f"**평점:** {row['AVRG_SCORE_VALUE']}")
                    if "positive_ratio" in row and not pd.isna(row["positive_ratio"]):
                        st.write(f"**긍정 비율:** {row['positive_ratio']:.2f}")
                    if "PEAK_SEASON" in row and not pd.isna(row["PEAK_SEASON"]):
                        season_korean = {v: k for k, v in season_mapping.items()}.get(
                            row["PEAK_SEASON"], row["PEAK_SEASON"]
                        )
                        st.write(f"**성수기:** {season_korean}")

                with col2:
                    # 월별 방문 비율 차트
                    try:
                        monthly_columns = [f"MONTH_{i+1}_RATIO" for i in range(12)]
                        existing_monthly_columns = [
                            col for col in monthly_columns if col in row.index
                        ]

                        if existing_monthly_columns:
                            monthly_data = [
                                row[col] for col in existing_monthly_columns
                            ]
                            month_labels = [
                                months[int(col.split("_")[1])]
                                for col in existing_monthly_columns
                            ]
                            st.bar_chart(
                                pd.DataFrame(
                                    {"방문 비율": monthly_data}, index=month_labels
                                )
                            )
                        else:
                            st.info("월별 방문 데이터가 없습니다.")
                    except Exception as e:
                        st.error(f"차트 표시 중 오류 발생: {e}")
                        st.info("월별 방문 데이터를 표시할 수 없습니다.")

                # 키워드 표시
                if (
                    "CORE_KWRD_CN" in row
                    and row["CORE_KWRD_CN"]
                    and not pd.isna(row["CORE_KWRD_CN"])
                ):
                    st.write("**주요 키워드:**")
                    keywords = row["CORE_KWRD_CN"].split(",")
                    st.write(", ".join(keywords))

elif recommendation_type == "맞춤 추천":
    st.subheader("여행 선호도 설정")

    # 사용자 선호도 입력
    col1, col2 = st.columns(2)

    with col1:
        user_region = st.selectbox("선호 지역", regions, key="user_region")
        user_category = st.selectbox("선호 카테고리", categories, key="user_category")
        user_season = st.selectbox("선호 계절", season_options, key="user_season")

    with col2:
        user_weekend = st.selectbox(
            "주중/주말 선호", weekend_options, key="user_weekend"
        )
        user_keywords = st.text_input("관심 키워드 (쉼표로 구분)", key="user_keywords")

    # 추천 버튼
    if st.button("맞춤 추천 받기", key="recommend_button"):
        # 필터링
        filtered_data = merged_data.copy()

        # 지역 필터링
        if user_region != "전체":
            filtered_data = filtered_data[filtered_data["지역"] == user_region]

        # 카테고리 필터링
        if user_category != "전체":
            filtered_data = filtered_data[filtered_data["CL_NM"] == user_category]

        # 계절 필터링
        if user_season != "전체" and "PEAK_SEASON" in filtered_data.columns:
            season_code = season_mapping.get(user_season)
            if season_code:
                filtered_data = filtered_data[
                    filtered_data["PEAK_SEASON"] == season_code
                ]

        # 주중/주말 필터링
        if user_weekend != "전체" and "WEEKEND_PREFERENCE" in filtered_data.columns:
            if user_weekend == "주중 선호":
                filtered_data = filtered_data[filtered_data["WEEKEND_PREFERENCE"] < 0.9]
            elif user_weekend == "주말 선호":
                filtered_data = filtered_data[filtered_data["WEEKEND_PREFERENCE"] > 1.1]
            elif user_weekend == "균형적":
                filtered_data = filtered_data[
                    (filtered_data["WEEKEND_PREFERENCE"] >= 0.9)
                    & (filtered_data["WEEKEND_PREFERENCE"] <= 1.1)
                ]

        # 키워드 필터링
        if user_keywords:
            keywords = [k.strip() for k in user_keywords.split(",")]
            keyword_match = filtered_data["features"].apply(
                lambda x: any(k.lower() in x.lower() for k in keywords)
            )
            filtered_data = filtered_data[keyword_match]

        # 결과 정렬 및 표시
        if len(filtered_data) > 0:
            # 인기도 순으로 정렬
            recommended_spots = filtered_data.sort_values(
                "popularity_score", ascending=False
            ).head(5)

            st.subheader("맞춤 추천 관광지")
            for i, (idx, row) in enumerate(recommended_spots.iterrows()):
                with st.expander(f"{i+1}. {row['AREA_NM']} ({row['CL_NM']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**지역:** {row['지역']}")
                        st.write(f"**카테고리:** {row['CL_NM']}")
                        st.write(f"**주소:** {row['ADDR']}")
                        if "AVRG_SCORE_VALUE" in row and not pd.isna(
                            row["AVRG_SCORE_VALUE"]
                        ):
                            st.write(f"**평점:** {row['AVRG_SCORE_VALUE']}")
                        if "positive_ratio" in row and not pd.isna(
                            row["positive_ratio"]
                        ):
                            st.write(f"**긍정 비율:** {row['positive_ratio']:.2f}")
                        if "PEAK_SEASON" in row and not pd.isna(row["PEAK_SEASON"]):
                            season_korean = {
                                v: k for k, v in season_mapping.items()
                            }.get(row["PEAK_SEASON"], row["PEAK_SEASON"])
                            st.write(f"**성수기:** {season_korean}")
                        if "WEEKEND_PREFERENCE" in row and not pd.isna(
                            row["WEEKEND_PREFERENCE"]
                        ):
                            if row["WEEKEND_PREFERENCE"] > 1.1:
                                weekend_pref = "주말 선호"
                            elif row["WEEKEND_PREFERENCE"] < 0.9:
                                weekend_pref = "주중 선호"
                            else:
                                weekend_pref = "균형적"
                            st.write(f"**주중/주말:** {weekend_pref}")

                    with col2:
                        # 월별 방문 비율 차트
                        try:
                            monthly_columns = [f"MONTH_{i+1}_RATIO" for i in range(12)]
                            existing_monthly_columns = [
                                col for col in monthly_columns if col in row.index
                            ]

                            if existing_monthly_columns:
                                monthly_data = [
                                    row[col] for col in existing_monthly_columns
                                ]
                                month_labels = [
                                    months[int(col.split("_")[1])]
                                    for col in existing_monthly_columns
                                ]
                                st.bar_chart(
                                    pd.DataFrame(
                                        {"방문 비율": monthly_data}, index=month_labels
                                    )
                                )
                            else:
                                st.info("월별 방문 데이터가 없습니다.")
                        except Exception as e:
                            st.error(f"차트 표시 중 오류 발생: {e}")
                            st.info("월별 방문 데이터를 표시할 수 없습니다.")

                        # 키워드 표시
                        if (
                            "CORE_KWRD_CN" in row
                            and row["CORE_KWRD_CN"]
                            and not pd.isna(row["CORE_KWRD_CN"])
                        ):
                            st.write("**주요 키워드:**")
                            keywords = row["CORE_KWRD_CN"].split(",")
                            st.write(", ".join(keywords))
        else:
            st.info("조건에 맞는 관광지가 없습니다. 선호도 설정을 변경해보세요.")

# 푸터
st.markdown("---")
st.markdown("© 제주도 관광지 추천 시스템")

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.plot(
        list(hourly_cols.values()),
        cluster_centers[i],
        marker="o",
        label=f"클러스터 {i}",
    )

plt.title("클러스터별 시간대 방문 패턴")
plt.xlabel("시간대")
plt.ylabel("방문 비율")
plt.legend()
plt.grid(True)
plt.show()
