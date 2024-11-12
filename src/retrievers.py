import streamlit as st
import torch
import os


def load_tensor_from_pkl(file_path):
    try:
        # 파일 경로가 존재하는지 확인
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")

        # pickle 파일을 torch.load()로 로드, map_location을 CPU로 지정
        with open(file_path, "rb") as f:
            tensor_data = torch.load(f, map_location=torch.device("cpu"))  # CPU로 로드

        return tensor_data

    except Exception as e:
        raise Exception(f"데이터 로드 중 오류 발생: {str(e)}")
