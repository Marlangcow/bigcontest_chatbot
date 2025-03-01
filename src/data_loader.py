import json
from typing import Dict, List, Any
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from src.config import JSON_PATHS


def load_documents():
    documents = {}
    for name, path in JSON_PATHS.items():
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            key_field = "가게명" if "mct" in name or "menu" in name else "관광지명"
            documents[name] = [
                Document(page_content=item[key_field], metadata=item) for item in data
            ]
        except Exception as e:
            print(f"파일 로드 중 오류 발생 ({path}): {str(e)}")
    return documents


def create_bm25_retriever(documents: List[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(documents)


def initialize_retrievers() -> Dict[str, BM25Retriever]:
    """각 데이터 소스에 대한 리트리버를 초기화합니다."""
    retrievers = {}
    documents = load_documents()

    for source_name, docs in documents.items():
        try:
            if docs:  # documents가 비어있지 않은 경우에만 처리
                retrievers[source_name] = create_bm25_retriever(docs)
        except Exception as e:
            print(f"리트리버 초기화 중 오류 발생 ({source_name}): {str(e)}")

    return retrievers
