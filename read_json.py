import json

# temp_data.json 파일 읽기
with open("temp_data.json", "r", encoding="utf-8") as f:
    collected_data = json.load(f)

# 수집된 데이터 출력
for item in collected_data:
    print(item)
