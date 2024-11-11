
import os
import pymysql
import json
import faiss
import torch
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Database connection setup
def create_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        db=os.getenv("DB_NAME"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

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

# Efficient document and FAISS index creation
def create_faiss_index(data, embedding_model, index_path):
    docs = [Document(page_content=item.get("가게명", ""), metadata=item) for item in data]
    db = FAISS.from_documents(docs, embedding_model)
    faiss.write_index(db.index, index_path)
    return db.as_retriever()

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
