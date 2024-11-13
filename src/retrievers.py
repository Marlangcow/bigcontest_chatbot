import streamlit as st
import pickle
import os

file_path = "./data/ensemble_retrievers_5.pkl"


def initialize_llm():
    # 예를 들어, Hugging Face의 특정 모델을 로드하는 경우
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"  # 사용할 모델 이름
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def load_retrievers_from_pkl(file_path):
    try:
        # 파일 경로가 존재하는지 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

        # pickle 파일을 CPU에서 로드
        with open(file_path, "rb") as f:
            retriever_data = pickle.load(f)  # pickle.load()로 로드

        return retriever_data

    except FileNotFoundError as fnf_error:
        st.error(f"파일을 찾을 수 없습니다: {str(fnf_error)}")
        return None
    except pickle.UnpicklingError as unpickling_error:
        st.error(
            f"Pickle 파일을 불러오는 데 문제가 발생했습니다: {str(unpickling_error)}"
        )
        return None
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None
