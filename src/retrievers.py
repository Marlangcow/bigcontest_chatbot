import json
import glob

# .json 파일만 가져오도록 필터링
retriever_file_paths = glob.glob("/Users/naeun/bigcontest_chatbot/data/*.json")


def load_retrievers_from_json():
    retriever_data = {}
    for file_path in retriever_file_paths:  # 중복된 인자를 제거
        print(f"로드 중: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                retrievers = json.load(file)
                retriever_data[file_path] = retrievers
                print(f"{file_path} 로드 성공")
        except Exception as e:
            print(f"파일 {file_path} 로드 중 오류 발생: {str(e)}")
    return retriever_data
