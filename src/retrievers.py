from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import util


def initialize_faiss_retrievers(docs, embedding):
    retrievers = {}
    for doc_type, doc_list in docs.items():
        # 문서 리스트가 비어있지 않은지 확인
        if not doc_list:
            print(f"Warning: Empty document list for {doc_type}")
            continue

        try:
            retriever = FAISS.from_documents(doc_list, embedding)
            retrievers[doc_type] = retriever.as_retriever()  # retriever 객체로 변환
        except Exception as e:
            print(f"Error creating FAISS retriever for {doc_type}: {str(e)}")
            continue

    return retrievers


def initialize_bm25_retrievers(docs):
    if not docs:  # docs가 비어있는지 확인
        raise ValueError("문서 리스트가 비어있습니다.")

    bm25_retrievers = {}
    for key, texts in docs.items():
        if not texts:  # 각 카테고리의 텍스트가 비어있는지 확인
            continue
        bm25_retrievers[key] = BM25Retriever.from_texts(texts)

    return bm25_retrievers


def initialize_ensemble_retrievers(retrievers, bm25_retrievers, weights=[0.6, 0.4]):
    ensemble_retrievers = {}
    for key in retrievers.keys():
        ensemble_retrievers[key] = EnsembleRetriever(
            retrievers=[retrievers[key], bm25_retrievers[key]], weights=weights
        )
    return ensemble_retrievers
