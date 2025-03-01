# src/recommender.py
import pandas as pd
import numpy as np
import os
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_collaborative_model(ratings_data):
    """협업 필터링 모델 학습"""
    print("협업 필터링 모델 학습 중...")

    # 모델이 이미 존재하는지 확인
    if os.path.exists(config.MODEL_PATH):
        print("이미 학습된 모델을 로드합니다.")
        with open(config.MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Surprise 데이터셋 생성
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_data[["user_id", "item_id", "rating"]], reader)

    # 학습 데이터셋 생성
    trainset = data.build_full_trainset()

    # SVD 모델 학습
    model = SVD(
        n_factors=config.SVD_PARAMS["n_factors"],
        n_epochs=config.SVD_PARAMS["n_epochs"],
        lr_all=config.SVD_PARAMS["lr_all"],
        reg_all=config.SVD_PARAMS["reg_all"],
    )
    model.fit(trainset)

    # 모델 저장
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def calculate_keyword_score(row, keywords):
    """키워드 매칭 점수 계산"""
    if "keywords" not in row or pd.isna(row["keywords"]) or row["keywords"] == "":
        return 0

    score = 0
    for keyword in keywords:
        if keyword.lower() in str(row["keywords"]).lower():
            score += 1
    return min(score / len(keywords) if keywords else 0, 1)  # 0~1 사이 값으로 정규화


def recommend_jeju_places(
    user_id,
    gender,
    is_tourist,
    region,
    min_rating=None,
    keywords=None,
    weekend_preference=None,
    season_preference=None,
    business_type=None,
    sales_group_preference=None,
):
    """하이브리드 추천 시스템"""
    print(f"사용자 {user_id}를 위한 추천 생성 중...")

    # 기본값 설정
    min_rating = min_rating or config.DEFAULT_MIN_RATING

    try:
        # 1. 데이터 로드
        if os.path.exists(config.FEATURES_PATH):
            features = pd.read_pickle(config.FEATURES_PATH)
        else:
            from src.data_processor import build_features_dataset

            features = build_features_dataset()

        if os.path.exists(config.RATINGS_PATH):
            ratings_data = pd.read_pickle(config.RATINGS_PATH)
        else:
            print("평점 데이터가 없습니다. 추천을 생성할 수 없습니다.")
            return pd.DataFrame()

        # 2. 협업 필터링 모델 로드 또는 학습
        model = train_collaborative_model(ratings_data)

        # 3. 지역 필터링
        if region:
            filtered_features = features[
                features["ADDR"].str.contains(region, na=False)
            ]
        else:
            filtered_features = features

        # 4. 업종 필터링 (있는 경우)
        if business_type and "CL_NM" in filtered_features.columns:
            filtered_features = filtered_features[
                filtered_features["CL_NM"].str.contains(business_type, na=False)
            ]

        # 5. 평점 필터링
        if min_rating > 0:
            filtered_features = filtered_features[
                filtered_features["rating"] >= min_rating
            ]

        # 6. 협업 필터링 점수 계산
        item_ids = filtered_features["AREA_NM"].unique()
        cf_predictions = []

        for item_id in item_ids:
            try:
                pred = model.predict(str(user_id), str(item_id))
                cf_predictions.append((item_id, pred.est))
            except Exception as e:
                print(f"예측 오류 ({item_id}): {e}")
                cf_predictions.append((item_id, 2.5))  # 기본값

        # 예측 결과를 데이터프레임으로 변환
        cf_df = pd.DataFrame(cf_predictions, columns=["AREA_NM", "cf_score"])

        # 7. 특성 데이터와 협업 필터링 결과 통합
        top_filtered = pd.merge(filtered_features, cf_df, on="AREA_NM", how="inner")

        # 8. 키워드 점수 계산 (있는 경우)
        if keywords and len(keywords) > 0:
            top_filtered["keyword_score"] = top_filtered.apply(
                lambda row: calculate_keyword_score(row, keywords), axis=1
            )
        else:
            top_filtered["keyword_score"] = 0.5  # 기본값

        # 9. 주말/주중 선호도 반영
        if weekend_preference is not None and "weekend_ratio" in top_filtered.columns:
            if weekend_preference:
                # 주말 선호하는 경우 주말 비율이 높은 곳 선호
                top_filtered["visit_pattern_score"] = top_filtered["weekend_ratio"]
            else:
                # 주중 선호하는 경우 주중 비율이 높은 곳 선호
                top_filtered["visit_pattern_score"] = 1 - top_filtered["weekend_ratio"]
        else:
            top_filtered["visit_pattern_score"] = 0.5

        # 10. 계절 선호도 반영
        if season_preference is not None:
            if season_preference == "여름" and "summer_ratio" in top_filtered.columns:
                top_filtered["season_score"] = top_filtered["summer_ratio"]
                # 여름 관광객 비율 반영 (있는 경우)
                if "summer_tourist_ratio" in top_filtered.columns:
                    tourist_factor = 1.2 if is_tourist else 0.8
                    top_filtered["season_score"] = top_filtered["season_score"] * (
                        top_filtered["summer_tourist_ratio"] * tourist_factor
                    )
            elif season_preference == "겨울" and "winter_ratio" in top_filtered.columns:
                top_filtered["season_score"] = top_filtered["winter_ratio"]
                # 겨울 관광객 비율 반영 (있는 경우)
                if "winter_tourist_ratio" in top_filtered.columns:
                    tourist_factor = 1.2 if is_tourist else 0.8
                    top_filtered["season_score"] = top_filtered["season_score"] * (
                        top_filtered["winter_tourist_ratio"] * tourist_factor
                    )
            elif (
                season_preference in ["봄", "가을"]
                and "spring_ratio" in top_filtered.columns
                and "fall_ratio" in top_filtered.columns
            ):
                if season_preference == "봄":
                    top_filtered["season_score"] = top_filtered["spring_ratio"]
                    # 봄 관광객 비율 반영 (있는 경우)
                    if "spring_tourist_ratio" in top_filtered.columns:
                        tourist_factor = 1.2 if is_tourist else 0.8
                        top_filtered["season_score"] = top_filtered["season_score"] * (
                            top_filtered["spring_tourist_ratio"] * tourist_factor
                        )
                else:
                    top_filtered["season_score"] = top_filtered["fall_ratio"]
                    # 가을 관광객 비율 반영 (있는 경우)
                    if "fall_tourist_ratio" in top_filtered.columns:
                        tourist_factor = 1.2 if is_tourist else 0.8
                        top_filtered["season_score"] = top_filtered["season_score"] * (
                            top_filtered["fall_tourist_ratio"] * tourist_factor
                        )
            else:
                top_filtered["season_score"] = 0.5
        else:
            top_filtered["season_score"] = 0.5

        # 11. 매출 그룹 선호도 반영 (있는 경우)
        if sales_group_preference is not None:
            sales_col = f"sales_group_{sales_group_preference}_tourist_ratio"
            if sales_col in top_filtered.columns:
                tourist_factor = 1.2 if is_tourist else 0.8
                top_filtered["sales_score"] = top_filtered[sales_col] * tourist_factor
            else:
                top_filtered["sales_score"] = 0.5
        else:
            top_filtered["sales_score"] = 0.5

        # 12. 관광객/현지인 선호도 반영
        if "avg_tourist_ratio" in top_filtered.columns:
            if is_tourist:
                # 관광객인 경우 관광객 비율이 높은 곳 선호
                top_filtered["tourist_score"] = top_filtered["avg_tourist_ratio"]
            else:
                # 현지인인 경우 현지인 비율이 높은 곳 선호
                top_filtered["tourist_score"] = 1 - top_filtered["avg_tourist_ratio"]
        else:
            top_filtered["tourist_score"] = 0.5

        # 13. 가중치 설정 (config에 추가 필요)
        weights = {
            "cf_score": config.WEIGHTS.get("cf_score", 0.3),
            "rating": config.WEIGHTS.get("rating", 0.2),
            "keyword": config.WEIGHTS.get("keyword", 0.2),
            "visit_pattern": config.WEIGHTS.get("visit_pattern", 0.1),
            "season": config.WEIGHTS.get("season", 0.1),
            "sales": 0.05,  # 새로운 가중치
            "tourist": 0.05,  # 새로운 가중치
        }

        # 가중치 정규화
        total_weight = sum(weights.values())
        for k in weights:
            weights[k] = weights[k] / total_weight

        # 14. 종합 점수 계산
        top_filtered["total_score"] = (
            top_filtered["cf_score"] / 5 * weights["cf_score"]
            + top_filtered.get("rating", 0) / 5 * weights["rating"]
            + top_filtered["keyword_score"] * weights["keyword"]
            + top_filtered["visit_pattern_score"] * weights["visit_pattern"]
            + top_filtered["season_score"] * weights["season"]
            + top_filtered["sales_score"] * weights["sales"]
            + top_filtered["tourist_score"] * weights["tourist"]
        )

        # 15. 최종 정렬
        top_filtered = top_filtered.sort_values("total_score", ascending=False)

        # 16. 결과 정리 (필요한 컬럼만 선택)
        result_columns = [
            "AREA_NM",
            "CL_NM",
            "ADDR",
            "rating",
            "review_count",
            "cf_score",
            "keyword_score",
            "visit_pattern_score",
            "season_score",
            "sales_score",
            "tourist_score",
            "total_score",
        ]

        if "keywords" in top_filtered.columns:
            result_columns.append("keywords")

        # 추가 정보 컬럼 (있는 경우)
        for col in ["avg_tourist_ratio", "MCT_TYPE", "BUSINESS_YEARS"]:
            if col in top_filtered.columns:
                result_columns.append(col)

        recommendations = top_filtered[result_columns].head(
            config.TOP_N_RECOMMENDATIONS
        )

        return recommendations

    except Exception as e:
        print(f"추천 생성 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()
