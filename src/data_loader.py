import json
from langchain.retrievers import BM25Retriever
from src.config import JSON_PATHS
from langchain.docstore.document import Document


def load_documents():
    documents = {}
    for name, path in JSON_PATHS.items():
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            # 필요한 필드만 선택적으로 로드
            key_field = "가게명" if "mct" in name or "menu" in name else "관광지명"
            documents[name] = [
                Document(page_content=item[key_field], metadata=item) for item in data
            ]
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {path}")
        except json.JSONDecodeError:
            print(f"JSON 디코딩 오류가 발생했습니다: {path}")
    return documents


# def load_documents():
#     documents = {}
#     for name, path in JSON_PATHS.items():
#         try:
#             with open(path, "r", encoding="utf-8") as file:
#                 data = json.load(file)
#             if name == "mct_json":
#                 documents[name] = [
#                     Document(page_content=item["가게명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#             elif name == "month_json":
#                 documents[name] = [
#                     Document(page_content=item["관광지명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#             elif name == "wkday_json":
#                 documents[name] = [
#                     Document(page_content=item["관광지명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#             elif name == "mop_sentiment_json":
#                 documents[name] = [
#                     Document(page_content=item["관광지명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#             elif name == "menu_json":
#                 documents[name] = [
#                     Document(page_content=item["가게명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#             elif name == "visit_jeju_json":
#                 documents[name] = [
#                     Document(page_content=item["관광지명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#             elif name == "kakaomap_reviews_json":
#                 documents[name] = [
#                     Document(page_content=item["관광지명"], metadata=item)
#                     for item in data  # 리스트로 처리
#                 ]
#         except FileNotFoundError:
#             print(f"파일을 찾을 수 없습니다: {path}")
#         except json.JSONDecodeError:
#             print(f"JSON 디코딩 오류가 발생했습니다: {path}")
#     return documents


def create_bm25_retriever(documents):
    if isinstance(documents, list) and all(isinstance(doc, str) for doc in documents):
        return BM25Retriever.from_texts(documents)
    return BM25Retriever.from_texts([doc.page_content for doc in documents])
