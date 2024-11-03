import os
from dotenv import load_dotenv
import pandas as pd
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# .env 파일의 환경 변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 API 키를 가져옵니다.
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
KEYWORD_LOCAL_URL = "https://dapi.kakao.com/v2/local/search/keyword.json?query={}"
MENU_URL = "https://place.map.kakao.com/main/v/{}"

headers = {
    "Authorization": f"KakaoAK {KAKAO_API_KEY}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 캐시 딕셔너리
cache = {}

def get_place_info_and_menus(place_name):
    if place_name in cache:
        return cache[place_name]

    search_query = f"제주 {place_name}"

    for attempt in range(3):  # 최대 3번 재시도
        try:
            response = requests.get(KEYWORD_LOCAL_URL.format(search_query), headers=headers)
            response.raise_for_status()
            data = response.json()
            documents = data.get('documents', [])

            if documents:
                place_id = documents[0].get('id')
                if place_id:
                    menu_response = requests.get(MENU_URL.format(place_id), headers=headers)
                    menu_response.raise_for_status()
                    menu_data = menu_response.json()['menuInfo'].get('menuList', [])

                    # 메뉴와 가격을 딕셔너리에 저장
                    menu_dict = {item.get('menu'): item.get('price') for item in menu_data}

                    cache[place_name] = menu_dict  # 가게명: 메뉴 딕셔너리 형태로 캐시 저장
                    return menu_dict

            return {}

        except requests.RequestException as e:
            if attempt == 2:  # 마지막 시도였다면
                print(f"Failed to process {place_name} after 3 attempts: {e}")
                return {}
            time.sleep(1)  # 재시도 전 1초 대기

def main(place_names):
    results = {}
    with ThreadPoolExecutor(max_workers=30) as executor:
        future_to_place = {executor.submit(get_place_info_and_menus, place_name): place_name for place_name in place_names}

        for future in tqdm(as_completed(future_to_place), total=len(place_names)):
            place_name = future_to_place[future]
            try:
                menus = future.result()
                if menus:  # 메뉴가 존재하는 경우에만 추가
                    results[place_name] = menus
            except Exception as exc:
                print(f'{place_name} generated an exception: {exc}')

    return results

if __name__ == "__main__":
    JEJU_MCT_GEO = pd.read_csv('mct.csv')  # CSV 파일 경로를 적절히 수정하세요
    place_names = JEJU_MCT_GEO['가게명'].unique().tolist()

    results = main(place_names)

    # 결과를 JSON 형식으로 저장
    with open('mct_menus.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("크롤링 완료. 결과가 'mct_menus.json'에 저장되었습니다.")
