import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

file_url = st.secrets["google_drive"]["csv_url"]


# 🔹 데이터 로드
@st.cache_resource
def load_data():
    df = pd.read_csv(file_url)

    # 중복된 가맹점 제거 (같은 가게가 여러 번 등장하는 문제 해결)
    df_unique = df.groupby("MCT_NM").first().reset_index()

    # 🔹 TF-IDF 벡터화
    vectorizer = TfidfVectorizer(max_features=5000)  # 5,000개의 주요 단어만 사용
    tfidf_matrix = vectorizer.fit_transform(df_unique["features"])

    # 🔹 KNN(Nearest Neighbors) 모델 구축
    knn_model = NearestNeighbors(
        n_neighbors=6, metric="cosine", algorithm="brute", n_jobs=-1
    )
    knn_model.fit(tfidf_matrix)

    # 🔹 가맹점 인덱스 매핑
    indices = pd.Series(df_unique.index, index=df_unique["MCT_NM"]).drop_duplicates()

    return df_unique, vectorizer, tfidf_matrix, knn_model, indices


# 데이터 로드
df_unique, vectorizer, tfidf_matrix, knn_model, indices = load_data()


def recommend_by_store(store_name, num_recommendations=5):
    """특정 가맹점과 유사한 가맹점 추천"""
    if store_name not in indices:
        return "❌ 유사한 식당이 없습니다."

    idx = indices[store_name]
    distances, indices_knn = knn_model.kneighbors(
        tfidf_matrix[idx], n_neighbors=num_recommendations + 1
    )

    store_indices = indices_knn.flatten()[1:]  # 자기 자신 제외
    return df_unique.iloc[store_indices][
        ["MCT_NM", "MCT_TYPE", "ADDR", "popularity_score"]
    ]


def recommend_by_keyword(keyword, num_recommendations=5):
    """
    특정 키워드(업종, 지역, 계절) 기반 추천
    - 1차 필터링: 키워드가 포함된 관광지 검색 (직접 검색)
    - 2차 추천: TF-IDF를 활용한 유사도 기반 추천 (키워드와 직접 매칭되지 않는 경우)
    """
    # 1️⃣ 키워드가 포함된 데이터 필터링
    keyword_filtered = df_unique[
        df_unique["features"].str.contains(keyword, case=False, na=False)
    ]

    # 2️⃣ 만약 직접 검색된 데이터가 충분하다면 이를 반환
    if len(keyword_filtered) >= num_recommendations:
        return keyword_filtered[
            ["MCT_NM", "MCT_TYPE", "ADDR", "popularity_score"]
        ].head(num_recommendations)

    # 3️⃣ TF-IDF 기반 검색 (키워드가 직접 포함되지 않은 경우)
    keyword_vector = vectorizer.transform([keyword])  # 키워드 벡터화
    distances, indices_knn = knn_model.kneighbors(
        keyword_vector, n_neighbors=num_recommendations
    )

    # 4️⃣ 추천된 데이터 반환
    store_indices = indices_knn.flatten()
    return df_unique.iloc[store_indices][
        ["MCT_NM", "MCT_TYPE", "ADDR", "popularity_score"]
    ]


def recommend_popular(num_recommendations=5):
    """인기도 기반 추천 (중복 제거 반영)"""
    return (
        df_unique[["MCT_NM", "MCT_TYPE", "ADDR", "popularity_score"]]
        .sort_values(by="popularity_score", ascending=False)
        .head(num_recommendations)
    )


def recommend_by_user_preference(
    gender=None, age_group=None, peak_time=None, num_recommendations=5
):
    """사용자 맞춤형 추천 (중복 제거 적용)"""
    filtered_df = df_unique.copy()

    if gender:
        filtered_df = filtered_df[filtered_df["MAIN_GENDER"] == gender]

    if age_group:
        filtered_df = filtered_df[filtered_df["MAIN_AGE_GROUP"] == age_group]

    if peak_time:
        filtered_df = filtered_df[filtered_df["PEAK_TIME"] == peak_time]

    return (
        filtered_df[["MCT_NM", "MCT_TYPE", "ADDR", "popularity_score"]]
        .sort_values(by="popularity_score", ascending=False)
        .head(num_recommendations)
    )
