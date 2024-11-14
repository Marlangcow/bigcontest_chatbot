import os
import torch
from sentence_transformers import util
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import StrOutputParser
import google.generativeai as genai

# 원본 GPU FAISS 인덱스 파일 경로
INDEX_PATHS = {
    "mct_db_index": "/content/mct_db_index.faiss",
    "month_db_index": "/content/month_db_index.faiss",
    "wkday_db_index": "/content/wkday_db_index.faiss",
    "mop_db_index": "/content/mob_db_index.faiss",  # mop_db_index는 flatL2 방식
    "menu_db_index": "/content/menus_db_index.faiss",
    "visit_jeju_db_index": "/content/visit_db_index.faiss",
    "kakaomap_reviews_db_index": "/content/kakaomap_reviews_index.faiss",
}


def initialize_mmr_retriever(db):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.6,
            "score_threshold": 0.8,
        },
    )


def initialize_ensemble_retriever_pair(retriever, bm25_retriever):
    return EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.6, 0.4])


# 데이터베이스를 검색기로 사용하기 위해 retriever 변수에 할당
mct_retriever = mct_db_index.as_retriever()
month_retriever = month_db_index.as_retriever()
wkday_retriever = wkday_db_index.as_retriever()
mop_retriever = mop_db_index.as_retriever()
mct_menus_retriever = menu_db_index.as_retriever()
visit_retriever = visit_jeju_db_index.sas_retriever()
kakaomap_reviews_retriever = kakaomap_reviews_db_index.as_retriever()


def initialize_retriever(
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


# 리스트로 DB와 이름을 묶어서 처리
dbs = {
    "mct": mct_db,
    "month": month_db,
    "wkday": wkday_db,
    "mop": mop_db,
    "menus": menus_db,
    "visit": visit_db,
    "kakaomap_reviews": kakaomap_reviews_db,
}

# 각 DB에 대해 리트리버 초기화
retrievers = {name: initialize_retriever(db) for name, db in dbs.items()}

# BM25 검색기 생성
mct_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in mct_docs])
month_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in month_docs]
)
wkday_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in wkday_docs]
)
mop_bm25_retriever = BM25Retriever.from_texts([doc.page_content for doc in mop_docs])
mct_menus_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in menu_docs]
)
visit_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in visit_docs]
)
kakaomap_reviews_bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in kakaomap_reviews_docs]
)


def initialize_ensemble_retriever(retrievers, weights):
    return EnsembleRetriever(retrievers=retrievers, weights=weights)


# 각 DB에 대해 리트리버와 BM25 리트리버 리스트를 묶은 딕셔너리
ensemble_retrievers = {
    "mct": (mct_retriever, mct_bm25_retriever),
    "mct_menus": (mct_menus_retriever, mct_menus_bm25_retriever),
    "mop": (mop_retriever, mop_bm25_retriever),
    "month": (month_retriever, month_bm25_retriever),
    "visit": (visit_retriever, visit_bm25_retriever),
    "wkday": (wkday_retriever, wkday_bm25_retriever),
    "kakaomap_reviews": (kakaomap_reviews_retriever, kakaomap_reviews_bm25_retriever),
}

# 앙상블 검색기 초기화
ensemble_retriever_objects = {
    name: initialize_ensemble_retriever(retrievers=retriever_pair, weights=[0.6, 0.4])
    for name, retriever_pair in ensemble_retrievers.items()
}

# Google Cloud Platform 프로젝트 ID 설정
os.environ["GOOGLE_CLOUD_PROJECT"] = "jeju-chatbot"

# API 키 설정 (선택 사항)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDY8hd_bq0yxBXxf1dv7okUVV9SX89vkyQ"

# Google Generative AI API 설정
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,  # 더 낮은 temperature로 설정해 할루시네이션 줄임
    top_p=0.85,  # top_p를 조정해 더 예측 가능한 답변 생성
    frequency_penalty=0.1,  # 같은 단어의 반복을 줄이기 위해 패널티 추가
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# 유연한 펑션 콜링을 위한 검색 및 병합 함수
def flexible_function_call_search(query):
    input_embedding = embedding.embed_query(query)  # 입력 쿼리를 임베딩

    retriever_mappings = {
        "mct": mct_ensemble_retriever,
        "mct_menus": mct_menus_ensemble_retriever,
        "mop": mop_ensemble_retriever,
        "month": month_ensemble_retriever,
        "visit": visit_ensemble_retriever,
        "wkday": wkday_ensemble_retriever,
        "kakaomap_reviews": kakaomap_reviews_ensemble_retriever,
    }

    retriever_descriptions = {
        "mct": "식당명 및 이용 비중 및 금액 비중",
        "mct_menus": "식당명 및 메뉴 및 금액",
        "mop": "관광지 전체 키워드 분석 데이터",
        "month": "관광지 월별 조회수",
        "visit": "관광지 핵심 키워드 및 정보",
        "wkday": "주별 일별 조회수 및 연령별 성별별 선호도",
        "kakaomap_reviews": "리뷰 데이터",
    }

    retriever_embeddings = {
        key: embedding.embed_query(value)
        for key, value in retriever_descriptions.items()
    }
    similarities = {
        key: util.cos_sim(input_embedding, torch.tensor(embed)).item()
        for key, embed in retriever_embeddings.items()
    }

    selected_retrievers = [key for key, sim in similarities.items() if sim >= 0.5]
    if not selected_retrievers:
        selected_retrievers = [max(similarities, key=similarities.get)]

    combined_results = {}
    for retriever in selected_retrievers:
        search_result = retriever_mappings[retriever].invoke(query)
        combined_results[retriever] = search_result

    merged_results = []
    for key, docs in combined_results.items():
        for doc in docs:
            if doc.page_content not in [
                result.page_content for result in merged_results
            ]:
                merged_results.append(doc)

    return merged_results


prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""
    ### 역할
    당신은 제주도 맛집과 관광지 추천 전문가입니다. 질문을 받을 때 논리적으로 생각한 후 단계별로 답변을 제공합니다.
    복잡한 질문일수록 천천히 생각하고 검색된 데이터를 바탕으로 친근하고 정겨운 답변을 제공합니다.

    ### Chain of Thought 방식 적용:
    1. 사용자의 질문을 단계별로 분석합니다.
    2. 질문의 위치 정보를 파악합니다.
    3. 그 후에 사용자가 제공한 정보나 검색된 데이터를 바탕으로 관련성 있는 맛집과 관광지를 추천합니다.
    4. 단계를 나누어 정보를 체계적으로 제공합니다.

    ### 지시사항
    1. 검색할 내용이 충분하지 않다면 사용자에게 반문하세요. 이는 가장 중요합니다. 단, 두번 이상 반문하지 마세요. 만약 사용자가 위치를 모른다면 제일 평점이 좋은 3개의 식당+카페와 3개의 관광지를 안내해주세요.
    2. 답변을 하는 경우 어떤 문서를 인용했는지 (키:값) 에서 키는 제외하고 값만 답변 뒤에 언급하세요.
      (mct_docs: 신한카드 가맹점 - 요식업, month_docs: 비짓제주 - 월별 조회수, wkday_docs: 비짓제주 - 요일별 조회수, mop_docs: 관광지 평점리뷰, menu_docs: 카카오맵 가게 메뉴, visit_docs: 비짓제주 - 여행지 정보, kakaomap_reviews_docs: 카카오맵 리뷰)
    4. 추천 이유와 거리, 소요 시간, 핵심키워드 3개, 평점과 리뷰들도 보여주세요. 만약 리뷰가 없는 곳이라면 ("아직 작성된 리뷰가 없습니다.") 라고 해주세요.
    5. 4번의 지시사항과 함께 판매 메뉴 2개, 가격을 함께 알려주세요.
    6. 주소를 바탕으로 실제 검색되는 장소를 아래 예시 링크 형식으로 답변하세요.
      - 네이버 지도 확인하기: (https://map.naver.com/p/search/제주도+<place>장소명</place>)
    7. 실제로 존재하는 식당과 관광지명을 추천해주어야 하며, %%흑돼지 맛집, 횟집 1 등 가게명이 명확하지 않은 답변은 하지 말아주세요.
    8. 답변 내용에 따라 폰트사이즈, 불렛, 순서를 활용하고 문단을 구분하여 가독성이 좋게 해주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """,
)

# 체인 생성
chain = LLMChain(
    prompt=prompt_template,
    llm=llm,
    output_parser=StrOutputParser(),
)


# 챗봇 대화 루프
def chat():
    print("챗봇 대화를 시작합니다. 'exit'을 입력하면 종료됩니다.")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    while True:
        user_input = input("질문을 입력하세요: ")
        if user_input.lower() == "exit":
            break

        search_results = flexible_function_call_search(user_input)
        search_results_str = "\n".join([doc.page_content for doc in search_results])
        chat_history = memory.load_memory_variables({})["chat_history"]

        input_data = {
            "input_text": user_input,
            "search_results": search_results_str,
            "chat_history": chat_history,
        }

        output = chain(input_data)
        output_text = output.get("text", str(output))

        print("\n챗봇 응답:", output_text)
        memory.save_context({"input": user_input}, {"output": output_text})


# 대화 실행
chat()
