# config.py
import os
from pathlib import Path
from typing import Dict, Any
import yaml

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
STATIC_DIR = BASE_DIR / ".static"
LOG_DIR = BASE_DIR / "logs"

# 데이터 파일 경로
MCT_DATA_PATH = RAW_DATA_DIR / "JEJU_MCT_DATA_v2.csv"

# 앱 설정
APP_CONFIG = {
    "title": "제주 맛집 추천 시스템",
    "description": "제주도의 맛집을 AI가 추천해드립니다",
    "theme": "light",
    "debug": False,
}

# API 설정
API_CONFIG = {"version": "v1", "base_url": "http://localhost:8000", "timeout": 30}

# 모델 파라미터
MODEL_CONFIG = {
    "svd": {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    "lightgbm": {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100},
}

# 추천 시스템 설정
RECOMMENDATION_CONFIG = {
    "min_rating": 3.5,
    "top_n": 10,
    "weights": {
        "cf_score": 0.4,
        "rating": 0.2,
        "keyword": 0.2,
        "visit_pattern": 0.1,
        "season": 0.1,
    },
}

# 캐시 설정
CACHE_CONFIG = {"enabled": True, "ttl": 3600, "max_size": 1000}  # 1시간

# 로깅 설정
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"}
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOG_DIR / "app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {"handlers": ["default", "file"], "level": "INFO", "propagate": True}
    },
}

# 필요한 디렉토리 생성
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """외부 설정 파일을 로드합니다."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# 환경변수에서 외부 설정 파일 경로를 가져옵니다
CONFIG_PATH = os.getenv("APP_CONFIG_PATH")
if CONFIG_PATH:
    external_config = load_config(CONFIG_PATH)
    # 외부 설정으로 기본 설정을 업데이트
    # TODO: 설정 병합 로직 구현
