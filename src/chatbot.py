from langchain.memory import ConversationBufferMemory
from typing import List
from langchain.schema import Document
import torch
from sentence_transformers import SentenceTransformer

# 임베딩 모델 초기화
embedding = SentenceTransformer("jhgan/ko-sroberta-multitask")


def get_chatbot_response(user_input, memory, chain, retrievers):
    search_results = flexible_function_call_search(user_input, retrievers)
    search_results_str = "\n".join([doc.page_content for doc in search_results]).strip()

    if not search_results_str:
        return "검색된 내용이 없어서 답변을 드릴 수 없습니다."

    chat_history = memory.load_memory_variables({}).get("chat_history", "")

    # 세션 상태에서 사용자 선택 값 가져오기
    keyword = st.session_state.get("keywords", "일반")
    location = st.session_state.get("locations", "제주시내")
    min_score = st.session_state.get("score", 4.5)

    input_data = {
        "input_text": user_input,
        "search_results": search_results_str,
        "chat_history": chat_history,
        "min_score": min_score,
        "location": location,
        "keyword": keyword,
    }

    try:
        output = chain(input_data)
        output_text = output.get("text", str(output))
    except Exception as e:
        print(f"LLM 응답 생성 중 오류 발생: {e}")
        return "응답을 생성하는 과정에서 오류가 발생했습니다. 다시 시도해주세요."

    memory.save_context({"input": user_input}, {"output": output_text})
    return output_text


# flexible_function_call_search 함수에 retrievers 인자 추가
def flexible_function_call_search(query, retrievers):
    input_embedding = embedding.encode(query)  # 입력 쿼리 임베딩

    # 리트리버 데이터 로드
    retriever_descriptions = {
        "mct": "식당명 및 이용 비중 및 금액 비중",
        "mct_menus": "식당명 및 메뉴 및 금액",
        "mop": "관광지 전체 키워드 분석 데이터",
        "month": "관광지 월별 조회수",
        "visit": "관광지 핵심 키워드 및 정보",
        "wkday": "주별 일별 조회수 및 연령별 성별별 선호도",
        "kakaomap_reviews": "리뷰 데이터",
    }

    # 각 리트리버 설명에 대한 임베딩 계산
    retriever_embeddings = {
        key: embedding.embed_query(value)
        for key, value in retriever_descriptions.items()
    }
    similarities = {
        key: util.cos_sim(input_embedding, torch.tensor(embed)).item()
        for key, embed in retriever_embeddings.items()
    }

    # 유사도가 0.5 이상인 리트리버 선택
    selected_retrievers = [key for key, sim in similarities.items() if sim >= 0.5]
    if not selected_retrievers:
        selected_retrievers = [
            max(similarities, key=similarities.get)
        ]  # 유사도가 0.5 미만인 경우 가장 유사한 리트리버 선택

    combined_results = {}
    for retriever_key in selected_retrievers:
        retriever = retrievers.get(retriever_key)
        if retriever:
            search_result = retriever.invoke(query)
            combined_results[retriever_key] = search_result

    # 결과 병합 및 응용 로직
    merged_results = []
    for key, docs in combined_results.items():
        for doc in docs:
            if doc.page_content not in [
                result.page_content for result in merged_results
            ]:
                merged_results.append(doc)

    return merged_results
