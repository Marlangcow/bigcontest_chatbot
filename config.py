# config.py
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# 파일 경로
WEEKLY_DATA_PATH = os.path.join(
    RAW_DATA_DIR, "제주 관광수요 예측 데이터_비짓제주 요일별 데이터/combined_wk.csv"
)
MONTHLY_DATA_PATH = os.path.join(
    RAW_DATA_DIR, "제주 관광수요 예측 데이터_비짓제주 월별 데이터/combined_mt.csv"
)
SENTIMENT_DATA_PATH = os.path.join(
    RAW_DATA_DIR, "제주 관광지 평점리뷰 감성분석 데이터/combined_sentiment.csv"
)
MORPHEME_DATA_PATH = os.path.join(
    RAW_DATA_DIR, "제주 관광지 평점리뷰 형태소 데이터/combined_mop.csv"
)
MCT_DATA_PATH = os.path.join(RAW_DATA_DIR, "JEJU_MCT_DATA_v2.csv")

# 처리된 데이터 경로
FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "jeju_place_features.pkl")
RATINGS_PATH = os.path.join(PROCESSED_DATA_DIR, "ratings_data.pkl")

# 모델 경로
MODEL_PATH = os.path.join(MODELS_DIR, "svd_model.pkl")

# 청크 크기 설정
CHUNK_SIZE = 10000

# 모델 파라미터
SVD_PARAMS = {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02}

# 추천 파라미터
DEFAULT_MIN_RATING = 3.5
TOP_N_RECOMMENDATIONS = 10

# 가중치 설정
WEIGHTS = {
    "cf_score": 0.4,
    "rating": 0.2,
    "keyword": 0.2,
    "visit_pattern": 0.1,
    "season": 0.1,
}

# 디렉토리 생성
for directory in [PROCESSED_DATA_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
