from src.data_loader import load_documents, create_bm25_retriever
from src.config import INDEX_PATHS
from langchain.retrievers import EnsembleRetriever
from faiss import read_index
import faiss
from sentence_transformers import SentenceTransformer

# 임베딩 모델 초기화
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return embedding_model


def convert_query_to_vector(query):
    # 쿼리를 임베딩 벡터로 변환
    model = get_embedding_model()
    return model.encode(query).reshape(1, -1)


def initialize_faiss_retrievers(index_paths):
    faiss_retrievers = {}
    for name, path in index_paths.items():
        index = faiss.read_index(path)
        faiss_retrievers[name] = lambda query: search_faiss_index(index, query)
    return faiss_retrievers


def search_faiss_index(index, query, k=10):
    query_vector = convert_query_to_vector(query)
    distances, indices = index.search(query_vector, k)
    return process_search_results(distances, indices)


def convert_query_to_vector(query):
    # 쿼리를 임베딩 벡터로 변환
    return embedding_model.encode(query).reshape(1, -1)


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


def initialize_retrievers(INDEX_PATHS):
    documents = load_documents()

    # INDEX_PATHS와 documents의 키를 일치시키기 위해 교집합을 사용
    common_keys = set(INDEX_PATHS.keys()).intersection(documents.keys())

    bm25_retrievers = {
        name: create_bm25_retriever(docs)
        for name, docs in documents.items()
        if name in common_keys
    }

    faiss_retrievers = initialize_faiss_retrievers(
        {name: INDEX_PATHS[name] for name in common_keys}
    )

    mmr_retrievers = {
        name: initialize_mmr_retriever(faiss_retrievers[name]) for name in common_keys
    }

    ensemble_retrievers = {
        name: initialize_ensemble_retriever(
            faiss_retrievers[name], bm25_retrievers[name], mmr_retrievers[name]
        )
        for name in common_keys
    }

    # retrievers를 딕셔너리로 반환
    return {
        "faiss": faiss_retrievers,
        "bm25": bm25_retrievers,
        "mmr": mmr_retrievers,
        "ensemble": ensemble_retrievers,
    }
