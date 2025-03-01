# src/data_processor.py
import pandas as pd
import numpy as np
import os
import pickle
from collections import Counter
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(file_path, encoding=None, chunksize=None):
    """데이터 로드 함수"""
    try:
        if encoding:
            return pd.read_csv(file_path, encoding=encoding, chunksize=chunksize)
        else:
            return pd.read_csv(file_path, chunksize=chunksize)
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return None


def process_weekly_data(weekly_data):
    """주간 데이터 처리"""
    print("주간 데이터 처리 중...")

    # 필요한 컬럼만 선택
    weekly_features = weekly_data[
        [
            "CL_CD",
            "CL_NM",
            "AREA_NM",
            "ADDR",
            "MON_VIEWS_CO",
            "TUES_VIEWS_CO",
            "WED_VIEWS_CO",
            "THUR_VIEWS_CO",
            "FRI_VIEWS_CO",
            "SAT_VIEWS_CO",
            "SUN_VIEWS_CO",
        ]
    ]

    # 주중/주말 비율 계산
    weekday_cols = [
        "MON_VIEWS_CO",
        "TUES_VIEWS_CO",
        "WED_VIEWS_CO",
        "THUR_VIEWS_CO",
        "FRI_VIEWS_CO",
    ]
    weekend_cols = ["SAT_VIEWS_CO", "SUN_VIEWS_CO"]
    all_days = weekday_cols + weekend_cols

    # 총 방문 수가 0인 경우 처리
    total_views = weekly_features[all_days].sum(axis=1)

    # 주중 비율 계산 (총 방문 수가 0인 경우 0.5로 설정)
    weekly_features["weekday_ratio"] = np.where(
        total_views > 0, weekly_features[weekday_cols].sum(axis=1) / total_views, 0.5
    )

    # 주말 비율 계산
    weekly_features["weekend_ratio"] = 1 - weekly_features["weekday_ratio"]

    # 요일별 선호도 패턴 (정규화)
    for day in all_days:
        weekly_features[f"{day}_norm"] = np.where(
            total_views > 0,
            weekly_features[day] / total_views,
            1 / 7,  # 데이터가 없으면 균등 분포 가정
        )

    return weekly_features


def process_monthly_data(monthly_data):
    """월별 데이터 처리"""
    print("월별 데이터 처리 중...")

    # 필요한 컬럼만 선택
    monthly_features = monthly_data[
        [
            "CL_CD",
            "CL_NM",
            "AREA_NM",
            "ADDR",
            "JAN_VIEWS_CO",
            "FEB_VIEWS_CO",
            "MAR_VIEWS_CO",
            "APR_VIEWS_CO",
            "MAY_VIEWS_CO",
            "JUN_VIEWS_CO",
            "JULY_VIEWS_CO",
            "AUG_VIEWS_CO",
            "SEP_VIEWS_CO",
            "OCT_VIEWS_CO",
            "NOV_VIEWS_CO",
            "DEC_VIEWS_CO",
        ]
    ]

    # 계절별 컬럼
    spring_cols = ["MAR_VIEWS_CO", "APR_VIEWS_CO", "MAY_VIEWS_CO"]
    summer_cols = ["JUN_VIEWS_CO", "JULY_VIEWS_CO", "AUG_VIEWS_CO"]
    fall_cols = ["SEP_VIEWS_CO", "OCT_VIEWS_CO", "NOV_VIEWS_CO"]
    winter_cols = ["DEC_VIEWS_CO", "JAN_VIEWS_CO", "FEB_VIEWS_CO"]
    all_months = spring_cols + summer_cols + fall_cols + winter_cols

    # 총 방문 수
    total_views = monthly_features[all_months].sum(axis=1)

    # 계절별 비율 계산
    monthly_features["spring_ratio"] = np.where(
        total_views > 0, monthly_features[spring_cols].sum(axis=1) / total_views, 0.25
    )
    monthly_features["summer_ratio"] = np.where(
        total_views > 0, monthly_features[summer_cols].sum(axis=1) / total_views, 0.25
    )
    monthly_features["fall_ratio"] = np.where(
        total_views > 0, monthly_features[fall_cols].sum(axis=1) / total_views, 0.25
    )
    monthly_features["winter_ratio"] = np.where(
        total_views > 0, monthly_features[winter_cols].sum(axis=1) / total_views, 0.25
    )

    # 월별 선호도 패턴 (정규화)
    for month in all_months:
        monthly_features[f"{month}_norm"] = np.where(
            total_views > 0,
            monthly_features[month] / total_views,
            1 / 12,  # 데이터가 없으면 균등 분포 가정
        )

    return monthly_features


def process_sentiment_data(sentiment_data):
    """감성 분석 데이터 처리"""
    print("감성 분석 데이터 처리 중...")

    # 필요한 컬럼만 선택
    if "reviewer_id" in sentiment_data.columns:
        sentiment_features = sentiment_data[
            ["CL_CD", "CL_NM", "AREA_NM", "rating", "sentiment_score"]
        ].copy()
    else:
        sentiment_features = sentiment_data[
            ["CL_CD", "CL_NM", "AREA_NM", "rating"]
        ].copy()
        sentiment_features["sentiment_score"] = np.nan

    # 장소별 평균 평점 및 감성 점수 계산
    sentiment_agg = sentiment_features.groupby(["CL_CD", "CL_NM", "AREA_NM"]).agg(
        rating=("rating", "mean"),
        sentiment_score=("sentiment_score", "mean"),
        review_count=("rating", "count"),
    )

    return sentiment_agg.reset_index()


def process_morpheme_data():
    """형태소 데이터 처리"""
    print("형태소 데이터 처리 중...")

    # 장소별 키워드 저장 딕셔너리
    place_keywords = {}

    # 청크 단위로 데이터 읽기
    for i, chunk in enumerate(
        load_data(config.MORPHEME_DATA_PATH, chunksize=config.CHUNK_SIZE)
    ):
        print(f"청크 {i+1} 처리 중...")

        # 각 장소별로 처리
        for (cl_cd, cl_nm, area_nm), group in chunk.groupby(
            ["CL_CD", "CL_NM", "AREA_NM"]
        ):
            key = (cl_cd, cl_nm, area_nm)

            # 이미 처리된 장소인 경우 키워드 추가
            if key in place_keywords:
                keywords = place_keywords[key]
            else:
                keywords = Counter()

            # 형태소 컬럼이 있는 경우에만 처리
            if "morpheme" in group.columns:
                # 각 리뷰의 형태소 처리
                for morpheme in group["morpheme"].dropna():
                    try:
                        # 형태소가 문자열인 경우 분할
                        if isinstance(morpheme, str):
                            words = morpheme.split()
                            keywords.update(words)
                    except Exception as e:
                        print(f"형태소 처리 오류: {e}")

            place_keywords[key] = keywords

    # 결과 데이터프레임 생성
    morpheme_features = []
    for (cl_cd, cl_nm, area_nm), keywords in place_keywords.items():
        # 상위 20개 키워드만 선택
        top_keywords = " ".join([k for k, _ in keywords.most_common(20)])
        morpheme_features.append(
            {
                "CL_CD": cl_cd,
                "CL_NM": cl_nm,
                "AREA_NM": area_nm,
                "keywords": top_keywords,
            }
        )

    return pd.DataFrame(morpheme_features)


def process_mct_data(mct_data):
    """카드 데이터 처리"""
    print("카드 데이터 처리 중...")

    # 필요한 컬럼만 선택
    mct_features = mct_data[
        [
            "MCT_NM",
            "MCT_TYPE",
            "TOURIST_RATIO",
            "BUSINESS_YEARS",
            "SEASON",
            "USE_AMOUNT_GROUP",
        ]
    ].copy()

    # 업장명을 AREA_NM과 매핑하기 위한 정규화
    mct_features["normalized_name"] = (
        mct_features["MCT_NM"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    )

    # 계절별 관광객 비율 피벗 테이블 생성
    season_pivot = pd.pivot_table(
        mct_features,
        values="TOURIST_RATIO",
        index="normalized_name",
        columns="SEASON",
        aggfunc="mean",
    ).reset_index()

    # 컬럼명 변경
    if "봄" in season_pivot.columns:
        season_pivot.rename(columns={"봄": "spring_tourist_ratio"}, inplace=True)
    if "여름" in season_pivot.columns:
        season_pivot.rename(columns={"여름": "summer_tourist_ratio"}, inplace=True)
    if "가을" in season_pivot.columns:
        season_pivot.rename(columns={"가을": "fall_tourist_ratio"}, inplace=True)
    if "겨울" in season_pivot.columns:
        season_pivot.rename(columns={"겨울": "winter_tourist_ratio"}, inplace=True)

    # 매출 그룹별 관광객 비율 피벗 테이블 생성
    sales_pivot = pd.pivot_table(
        mct_features,
        values="TOURIST_RATIO",
        index="normalized_name",
        columns="USE_AMOUNT_GROUP",
        aggfunc="mean",
    ).reset_index()

    # 컬럼명 변경
    for col in sales_pivot.columns:
        if col != "normalized_name":
            sales_pivot.rename(
                columns={col: f"sales_group_{col}_tourist_ratio"}, inplace=True
            )

    # 두 피벗 테이블 병합
    mct_processed = pd.merge(
        season_pivot, sales_pivot, on="normalized_name", how="outer"
    )

    # 평균 관광객 비율 계산
    mct_avg = (
        mct_features.groupby("normalized_name")["TOURIST_RATIO"].mean().reset_index()
    )
    mct_avg.rename(columns={"TOURIST_RATIO": "avg_tourist_ratio"}, inplace=True)

    # 최종 병합
    mct_processed = pd.merge(mct_processed, mct_avg, on="normalized_name", how="outer")

    return mct_processed


def prepare_ratings_data(sentiment_data):
    """평점 데이터 준비"""
    print("평점 데이터 준비 중...")

    # 리뷰어 ID가 없는 경우 리뷰 ID를 사용자 ID로 활용
    if "reviewer_id" not in sentiment_data.columns:
        ratings_data = sentiment_data[["review_id", "AREA_NM", "rating"]].copy()
        ratings_data.columns = ["user_id", "item_id", "rating"]
    else:
        ratings_data = sentiment_data[["reviewer_id", "AREA_NM", "rating"]].copy()
        ratings_data.columns = ["user_id", "item_id", "rating"]

    # 결측치 제거
    ratings_data = ratings_data.dropna()

    # 사용자 ID와 아이템 ID를 문자열로 변환
    ratings_data["user_id"] = ratings_data["user_id"].astype(str)
    ratings_data["item_id"] = ratings_data["item_id"].astype(str)

    # 평점 데이터 저장
    ratings_data.to_pickle(config.RATINGS_PATH)

    return ratings_data


def build_features_dataset():
    """특성 데이터셋 구축"""
    # 이미 처리된 데이터가 있는지 확인
    if os.path.exists(config.FEATURES_PATH):
        print("이미 처리된 특성 데이터를 로드합니다.")
        return pd.read_pickle(config.FEATURES_PATH)

    print("특성 데이터셋 구축 시작...")

    # 1. 주간 데이터 처리
    weekly_data = load_data(config.WEEKLY_DATA_PATH)
    weekly_features = process_weekly_data(weekly_data)
    del weekly_data  # 메모리 확보

    # 2. 월간 데이터 처리
    monthly_data = load_data(config.MONTHLY_DATA_PATH)
    monthly_features = process_monthly_data(monthly_data)
    del monthly_data  # 메모리 확보

    # 3. 감성 분석 데이터 처리
    sentiment_data = load_data(config.SENTIMENT_DATA_PATH)
    sentiment_features = process_sentiment_data(sentiment_data)

    # 4. 평점 데이터 준비
    ratings_data = prepare_ratings_data(sentiment_data)
    del sentiment_data  # 메모리 확보

    # 5. 형태소 데이터 처리
    morpheme_features = process_morpheme_data()

    # 6. 카드 데이터 처리
    try:
        mct_data = load_data(config.MCT_DATA_PATH, encoding="cp949")
        mct_features = process_mct_data(mct_data)
        del mct_data  # 메모리 확보
    except Exception as e:
        print(f"카드 데이터 처리 중 오류: {e}")
        mct_features = None

    # 7. 데이터 통합
    print("데이터 통합 중...")
    # 장소 정보를 키로 사용하여 데이터 통합
    features_combined = pd.merge(
        weekly_features,
        monthly_features,
        on=["CL_CD", "CL_NM", "AREA_NM", "ADDR"],
        how="outer",
    )

    features_combined = pd.merge(
        features_combined,
        sentiment_features,
        on=["CL_CD", "CL_NM", "AREA_NM"],
        how="left",
    )

    features_combined = pd.merge(
        features_combined,
        morpheme_features,
        on=["CL_CD", "CL_NM", "AREA_NM"],
        how="left",
    )

    # 카드 데이터와 통합 (있는 경우)
    if mct_features is not None:
        # AREA_NM을 정규화하여 매핑
        features_combined["normalized_name"] = (
            features_combined["AREA_NM"]
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
        )

        # 유사도 기반 매핑 또는 부분 매칭
        features_combined = pd.merge(
            features_combined, mct_features, on="normalized_name", how="left"
        )

    # 8. 결측치 처리
    features_combined["rating"] = features_combined["rating"].fillna(0)
    features_combined["review_count"] = features_combined["review_count"].fillna(0)

    # 관광객 비율 결측치 처리
    tourist_ratio_cols = [
        col for col in features_combined.columns if "tourist_ratio" in col
    ]
    for col in tourist_ratio_cols:
        features_combined[col] = features_combined[col].fillna(0.5)  # 중간값으로 설정

    # 9. 처리된 특성 저장
    print("처리된 특성 저장 중...")
    features_combined.to_pickle(config.FEATURES_PATH)

    print("특성 데이터셋 구축 완료!")
    return features_combined
