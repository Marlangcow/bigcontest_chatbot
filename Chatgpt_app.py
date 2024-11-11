import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import streamlit as st

from transformers import AutoTokenizer, AutoModel
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from google.cloud import dialogflow_v2 as dialogflow
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
import google.generativeai as genai
from typing import List, Dict
from langchain_community.embeddings import (
    SentenceTransformerEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import faiss


# Database connection
def create_db_connection():
    try:
        return pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            db=os.getenv("DB_NAME"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
        )
    except pymysql.Error as e:
        st.error(f"데이터베이스 연결 실패: {e}")
        return None


# Fetch data from database
def fetch_data(keyword):
    connection = create_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM your_table_name WHERE category=%s"
            cursor.execute(sql, (keyword,))
            result = cursor.fetchall()
        return result
    finally:
        connection.close()


# FAISS 인덱스 생성 함수에 예외 처리 추가
def create_faiss_index(data, embedding_model, index_path):
    try:
        docs = [
            Document(page_content=item.get("가게명", ""), metadata=item)
            for item in data
        ]
        db = FAISS.from_documents(docs, embedding_model)
        faiss.write_index(db.index, index_path)
        return db.as_retriever()
    except Exception as e:
        st.error(f"FAISS 인덱스 생성 실패: {e}")
        return None


# Load environment variables
load_dotenv()

# 환경 변수 검증 추가
required_env_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")

# Define paths and initialize retrievers
file_paths = {
    "mct": "path/to/mct.json",
    "month": "path/to/month.json",
    "wkday": "path/to/wkday.json",
    "menu": "path/to/menu.json",
    # Add more if necessary
}

embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
retrievers = {}

for key, path in file_paths.items():
    with open(path, "r") as f:
        data = json.load(f)
    retrievers[key] = create_faiss_index(data, embedding_model, f"{key}_index.faiss")

# Streamlit UI and testing database connection
st.title("🍊 감귤톡, 제주도 여행 메이트")
st.sidebar.selectbox("원하는 키워드", ["착한가격업소", "한식", "카페"])
