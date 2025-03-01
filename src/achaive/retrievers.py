import json
from src.data_loader import load_documents, create_bm25_retriever
from src.config import INDEX_PATHS
from langchain.retrievers import EnsembleRetriever
from faiss import read_index
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from src.utils.embedding import get_embedding_model


def convert_query_to_vector(query):
    model = get_embedding_model()
    return model.encode(query).reshape(1, -1)


def initialize_faiss_retrievers(index_paths):
    # FAISS 인덱스를 로드하고 검색 함수를 초기화
    faiss_retrievers = {}
    for name, path in index_paths.items():
        index = faiss.read_index(path)
        faiss_retrievers[name] = lambda query: search_faiss_index(index, query)
    return faiss_retrievers


def search_faiss_index(index, query, k=10):
    query_vector = convert_query_to_vector(query)
    distances, indices = index.search(query_vector, k)
    return process_search_results(distances, indices)


def process_search_results(distances, indices):
    # 검색 결과를 적절히 처리하여 반환
    # 예: Document 객체로 변환
    results = []
    for i in range(len(indices[0])):
        # 예시로, 인덱스와 거리를 사용하여 결과를 생성
        results.append({"index": indices[0][i], "distance": distances[0][i]})
    return results


def initialize_mmr_retriever(
    db, search_type="mmr", k=4, fetch_k=10, lambda_mult=0.6, score_threshold=0.8
):
    return db.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "score_threshold": score_threshold,
        },
    )


def initialize_ensemble_retriever(
    faiss_retriever, bm25_retriever, mmr_retriever, weights=[0.4, 0.3, 0.3]
):
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever, mmr_retriever], weights=weights
    )


def create_retriever_from_documents(documents: List[Document]) -> BM25Retriever:
    """문서 리스트로부터 BM25 리트리버를 생성합니다."""
    return BM25Retriever.from_documents(documents)


def process_json_data(data: List[Dict[str, Any]]) -> List[Document]:
    """JSON 데이터를 Document 객체 리스트로 변환합니다."""
    documents = []
    for item in data:
        # 관광지명 또는 가게명을 page_content로 사용
        content = item.get("관광지명") or item.get("가게명", "")
        if content:
            documents.append(Document(page_content=content, metadata=item))
    return documents


def initialize_retrievers(
    json_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, BM25Retriever]:
    """각 데이터 소스에 대한 리트리버를 초기화합니다."""
    retrievers = {}
    for source_name, data in json_data.items():
        try:
            documents = process_json_data(data)
            if documents:
                retrievers[source_name] = create_retriever_from_documents(documents)
        except Exception as e:
            print(f"리트리버 초기화 중 오류 발생 ({source_name}): {str(e)}")
    return retrievers


class ReviewRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.review_vectors = None
        self.reviews = None

    def fit(self, reviews):
        self.reviews = reviews
        self.review_vectors = self.vectorizer.fit_transform(reviews)

    def get_relevant_reviews(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.review_vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.reviews[i] for i in top_indices]
