import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('window-size=1920x1080')
driver = webdriver.Chrome(options=options)

# 수집할 기본 URL과 content_ids 목록
base_url = "https://visitjeju.net/kr/detail/view?contentsid="
content_ids = [
    "CONT_000000000500477",
    "CONT_000000000500281",
    "CONT_000000000500309",
    # 추가로 수집할 content_id를 여기에 추가
]

# 데이터를 저장할 리스트
data = []

# 각 URL에서 데이터 수집
for content_id in content_ids:
    url = f"{base_url}{content_id}&menuId=DOM_000001718000000000#p1"
    
    # 페이지 로드
    driver.get(url)
    time.sleep(5)  # 페이지 로딩 대기 시간

    # 현재 페이지의 HTML 소스를 가져오기
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # h3 요소 가져오기
    h3_element = soup.find('h3')
    h3_text = h3_element.text.strip() if h3_element else 'N/A'

    # class="tag_area"에서 best_tag와 다른 p 태그 가져오기
    tag_area_element = soup.find(class_='tag_area')
    best_tags = [a.text.strip() for a in tag_area_element.find_all('a')] if tag_area_element else []
    p_elements = tag_area_element.find_all('p') if tag_area_element else []
    p_texts = [p.text.strip() for p in p_elements if p.text.strip()]

    # class="info_sub_cont" 요소 가져오기
    info_elements = soup.find_all(class_='info_sub_cont')
    info_dict = {info.find_previous_sibling('p').text.strip(): info.text.strip() for info in info_elements}

    # 수집한 데이터 저장
    data.append({
        "Title": h3_text,
        "Best Tags": ', '.join(best_tags),
        "Additional P Texts": ', '.join(p_texts),
        **info_dict  # 기본 정보 추가
    })

# DataFrame으로 변환
df = pd.DataFrame(data)

# CSV로 저장
df.to_csv('jeju_data.csv', index=False, encoding='utf-8-sig')

# 드라이버 종료
driver.quit()

print("데이터 수집 완료! 'jeju_data.csv' 파일로 저장되었습니다.")
