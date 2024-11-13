import streamlit as st
import json
import glob
from langchain.retrievers import EnsembleRetriever

# 여러 .pkl 파일 경로
# retriever_file_paths = [
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/mct.pkl",
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/month.pkl",
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/wkday.pkl",
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/mop_sentiment.pkl",
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/menu.pkl",
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/visit_jeju.pkl",
#     "/Users/naeun/bigcontest_chatbot/data/retrievers/kakaomap_reviews.pkl",
# ]

# # pickle 파일 로드 함수 (EnsembleRetriever 객체 로드)
# def load_retrievers_from_pkl(file_paths):
#     retriever_data = {}

#     for file_path in file_paths:
#         print(f"로드 중: {file_path}")
#         try:
#             with open(file_path, "rb") as file:
#                 # pickle.load()로 객체를 로드
#                 retrievers = pickle.load(file)
#                 retriever_data[file_path] = retrievers
#                 print(f"{file_path} 로드 성공")
#         except Exception as e:
#             print(f"파일 {file_path} 로드 중 오류 발생: {str(e)}")

#     return retriever_data


# .json 파일만 가져오도록 필터링
retriever_file_paths = glob.glob(
    "/Users/naeun/bigcontest_chatbot/data/json_retrievers/*.json"
)


def load_retrievers_from_json(file_paths):
    retriever_data = {}
    for file_path in file_paths:
        print(f"로드 중: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                retrievers = json.load(file)
                retriever_data[file_path] = retrievers
                print(f"{file_path} 로드 성공")
        except Exception as e:
            print(f"파일 {file_path} 로드 중 오류 발생: {str(e)}")
    return retriever_data


# JSON에서 로드된 데이터를 바탕으로 객체 복원
def load_ensemble_retriever_from_json(load_retrievers_from_json):
    retriever_data = {}
    for file_path in retriever_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 'type'을 확인하고 해당 클래스를 복원할 로직 작성
            if (
                data["type"]
                == "<class 'langchain.retrievers.ensemble.EnsembleRetriever'>"
            ):
                # retrievers와 weights 데이터를 복원
                retrievers_data = eval(data["data"])  # 문자열을 파이썬 객체로 변환

                # retrievers와 weights를 분리하여 객체 생성
                retrievers = retrievers_data["retrievers"]
                weights = retrievers_data["weights"]

                # EnsembleRetriever 객체 생성
                ensemble_retriever = EnsembleRetriever(
                    retrievers=retrievers, weights=weights
                )
                retriever_data[file_path] = ensemble_retriever
                # st.write(f"{file_path} 로드 성공")  # 출력 제거
            else:
                raise ValueError(f"Unsupported type for restoration in {file_path}")
        except Exception as e:
            st.error(f"{file_path} 로드 중 오류 발생: {str(e)}")

    return retriever_data


# json 파일 로드 (한 번만 로드되므로 캐시 활용)
if "retriever_data" not in st.session_state:
    retriever_data = load_retrievers_from_json(retriever_file_paths)
    if retriever_data:
        st.session_state.retriever_data = retriever_data
        st.session_state.retrievers = retriever_data  # "retrievers" 키 초기화

        # 메시지를 일시적으로 표시
        message_placeholder = st.empty()
        message_placeholder.write("🌊🌊🌊잠시만 기다려주세요🏄🏄🏄 ")

        # 로드가 완료되면 메시지 제거
        message_placeholder.empty()
else:
    retriever_data = st.session_state.retriever_data
