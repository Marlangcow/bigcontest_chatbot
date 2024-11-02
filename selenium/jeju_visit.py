import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 드라이버 설정
options = Options()
options.add_argument('--headless')  # 브라우저 창 숨김
options.add_argument('window-size=1920x1080')  # 창 크기 설정
options.add_argument('--blink-settings=imagesEnabled=false')  # 이미지 로딩 비활성화
options.page_load_strategy = 'eager'

driver = webdriver.Chrome(options=options)

# 수집할 URL 설정
menu_ids = [
    "DOM_000001720000000000",  # 쇼핑 ID
    "DOM_000001718000000000",  # 관광지 ID
    "DOM_000001707000000000",  # 숙박 ID
    "DOM_000001719000000000"   # 식당 ID
]

category_nums = [
    "cate0000000003",   # 쇼핑
    "cate0000000002",  # 관광지
    "cate0000000004",  # 숙박
    "cate0000000005"  # 식당
]

base_url = "https://visitjeju.net/kr/detail/view?contentsid="

# 콘텐츠 ID를 저장할 리스트
content_ids = []
data = []  # 세부 데이터를 저장할 리스트
batch_size = 10  # 배치 크기 설정

# 콘텐츠 ID 수집
for menu_id, category_num in zip(menu_ids, category_nums):
    page_num = 1
    while True:
        print(f"수집 중: 메뉴 {menu_id}, 페이지 {page_num}")
        url = f"https://visitjeju.net/kr/detail/list?menuId={menu_id}&cate1cd={category_num}#p{page_num}"
        
        driver.get(url)
        time.sleep(3)  # 페이지 로딩 대기

        # 현재 페이지의 HTML 소스를 가져오기
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 콘텐츠 ID 수집
        page_content_ids = []
        content_links = soup.find_all('a', href=True)
        for link in content_links:
            if 'contentsid=' in link['href']:  
                content_id = link['href'].split('contentsid=')[1].split('&')[0]
                if content_id not in content_ids:
                    content_ids.append(content_id)
                    page_content_ids.append(content_id)
                    print(f"추출된 콘텐츠 ID: {content_id}")

        # 더 이상 콘텐츠 ID가 없다면 종료
        if not page_content_ids:
            print(f"메뉴 {menu_id}의 수집이 완료되었습니다.")
            break

        page_num += 1

# 수집한 콘텐츠 ID로 세부 정보 수집
for idx, content_id in enumerate(content_ids, start=1):
    url = f"{base_url}{content_id}&menuId={menu_id}"
    print(f"수집 중: {idx}/{len(content_ids)} - 콘텐츠 ID: {content_id}")

    driver.get(url)
    
    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, 'h3'))
        )
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # 제목 및 태그 영역, 추가 정보 수집
        h3_element = soup.find('h3')
        h3_text = h3_element.text.strip() if h3_element else "제목 없음"

        tag_area_element = soup.find(class_='tag_area')
        best_tags = [a.text.strip() for a in tag_area_element.find_all('a')] if tag_area_element else []
        p_elements = tag_area_element.find_all('p') if tag_area_element else []
        p_texts = [p.text.strip() for p in p_elements if p.text.strip()]
        
        info_elements = soup.find_all(class_='info_sub_cont')
        info_dict = {info.find_previous_sibling('p').text.strip(): info.text.strip() for info in info_elements}

        # 데이터 저장
        content_data = {
            "Content ID": content_id,
            "Title": h3_text,
            "Best Tags": ', '.join(best_tags),
            "Additional P Texts": ', '.join(p_texts),
            **info_dict
        }
        data.append(content_data)
        print(content_data)
        
        # batch_size 단위로 임시 저장
        if idx % batch_size == 0:
            temp_df = pd.DataFrame(data)
            temp_df.to_csv('temp_jeju_visit.csv', index=False, encoding='utf-8-sig')
            print(f"{idx}개 데이터까지 임시 저장 완료: 'temp_jeju_visit.csv'")
            
    except Exception as e:
        print(f"콘텐츠 ID {content_id}에서 데이터를 가져오지 못했습니다: {e}")

# 모든 콘텐츠 ID 수집 완료 후 최종 저장
df = pd.DataFrame(data)
df.to_csv('jeju_visit.csv', index=False, encoding='utf-8-sig')
driver.quit()
print("데이터 수집 완료! 'jeju_visit.csv' 파일로 저장되었습니다.")
