import json
import os
import faiss
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_json_files(FILE_PATHS):
    """
    주어진 파일 경로에서 JSON 파일들을 로드합니다.

    Args:
        file_paths (dict): JSON 파일들의 경로를 담은 딕셔너리
        {
            "mct": 'data/mct.json',
            "month": 'data/month.json',
            "wkday": 'data/wkday.json',
            "mop_sentiment": 'data/merge_mop_sentiment.json',
            "menu": 'data/mct_menus.json',
            "visit_jeju": 'data/visit_jeju.json',
            "kakaomap_reviews": 'data/kakaomap_reviews.json'
        }

    Returns:
        dict: 로드된 JSON 데이터를 담은 딕셔너리
    """
    data = {}
    for key, path in FILE_PATHS.items():
        try:
            with open(path, "r", encoding="utf-8") as file:
                data[key] = json.load(file)
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {path}")
            data[key] = []  # 파일이 없는 경우 빈 리스트 반환
        except json.JSONDecodeError:
            print(f"JSON 파일 파싱 오류: {path}")
            data[key] = []  # JSON 파싱 오류시 빈 리스트 반환
    return data


def load_faiss_indexes(INDEX_PATHS):
    indexes = {}
    for key, path in INDEX_PATHS.items():
        if os.path.exists(path):
            try:
                indexes[key] = faiss.read_index(path)
            except faiss.FaissException as e:
                print(f"Error loading index '{key}': {e}")
    return indexes


def create_documents(data):
    return {
        "mct_docs": [
            Document(page_content=item.get("가게명", ""), metadata=item)
            for item in data["mct"]
        ],
        "month_docs": [
            Document(page_content=item.get("관광지명", ""), metadata=item)
            for item in data["month"]
        ],
        "wkday_docs": [
            Document(page_content=item.get("관광지명", ""), metadata=item)
            for item in data["wkday"]
        ],
        "mop_docs": [
            Document(page_content=item.get("관광지명", ""), metadata=item)
            for item in data["mop_sentiment"]
        ],
        "menu_docs": [
            Document(page_content=item.get("가게명", ""), metadata=item)
            for item in data["menu"]
        ],
        "visit_docs": [
            Document(page_content=item.get("관광지명", ""), metadata=item)
            for item in data["visit_jeju"]
        ],
        "kakaomap_reviews_docs": [
            Document(page_content=item.get("관광지명", ""), metadata=item)
            for item in data["kakaomap_reviews"]
        ],
    }


def initialize_embeddings(model_name="jhgan/ko-sroberta-multitask"):
    """
    사용자 입력 임베딩을 위한 HuggingFaceEmbeddings만 초기화합니다.
    """
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    return embedding
