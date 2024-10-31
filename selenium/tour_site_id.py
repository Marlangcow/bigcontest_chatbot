import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 브라우저 창 숨김
options.add_argument('window-size=1920x1080')  # 창 크기 설정
driver = webdriver.Chrome(options=options)

# 수집할 기본 URL 및 페이지 수 설정
menu_id = "DOM_000001718000000000"  # 메뉴 ID
total_pages = 127  # 페이지 수 (1부터 127까지)

# 콘텐츠 ID를 저장할 리스트
content_ids = []

# 각 페이지 반복
for page_num in range(1, total_pages + 1):
    print(f"수집 중: 페이지 {page_num}/{total_pages}")  # 진행 상황 출력
    url = f"https://visitjeju.net/kr/detail/list?menuId={menu_id}&page={page_num}"

    # 드라이버로 URL 접근
    driver.get(url)
    time.sleep(5)  # 페이지 로딩 대기 (필요에 따라 조정)

    # 현재 페이지의 HTML 소스를 가져오기
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # 콘텐츠 ID 수집
    try:
        # 콘텐츠 링크 추출
        content_links = soup.find_all('a', href=True)  # 모든 링크를 찾음
        for link in content_links:
            if 'contentsid=' in link['href']:  # 링크에서 contentsid가 있는지 확인
                content_id = link['href'].split('contentsid=')[1].split('&')[0]  # 콘텐츠 ID 추출
                content_ids.append(content_id)
                print(f"추출된 콘텐츠 ID: {content_id}")

    except Exception as e:
        print(f"콘텐츠 ID 수집 중 오류 발생: {e}")

# 드라이버 종료
driver.quit()

# 수집된 콘텐츠 ID 개수 및 내용 출력
print(f"수집된 콘텐츠 ID 개수: {len(content_ids)}")
print(content_ids)

# DataFrame으로 변환 후 CSV로 저장
df = pd.DataFrame(content_ids, columns=["Content ID"])
df.to_csv('jeju_content_ids.csv', index=False, encoding='utf-8-sig')
print("콘텐츠 ID 수집 완료! 'jeju_content_ids.csv' 파일로 저장되었습니다.")
