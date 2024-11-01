import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup

# 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 브라우저 창 숨김
options.add_argument('window-size=1920x1080')  # 창 크기 설정
driver = webdriver.Chrome(options=options)

# 수집할 기본 URL 및 페이지 수 설정
base_url = "https://visitjeju.net/kr/detail/view?contentsid="
menu_id = "DOM_000001718000000000"  # 관광지 ID

# CSV 파일에서 contents_id 읽기
content_ids_df = pd.read_csv('tour_site_id.csv')
content_ids = content_ids_df['Content ID'].tolist()  # 'Content ID'라는 열에서 ID를 리스트로 변환

# 데이터를 저장할 리스트
data = []

# 각 콘텐츠 ID에 대해 데이터 수집
for idx, content_id in enumerate(content_ids, start=1):
    # 페이지 번호를 순회하여 데이터를 수집
    for page_num in range(1, 128):  # 페이지 1부터 127까지
        url = f"{base_url}{content_id}&menuId={menu_id}#p{page_num}"
        
        print(f"수집 중: {idx}/{len(content_ids)} - 콘텐츠 ID: {content_id} - 페이지: {page_num}")  # 진행 상황 출력
        
        # 페이지 로드
        driver.get(url)
        time.sleep(5)  # 페이지 로딩 대기 시간

        # 현재 페이지의 HTML 소스를 가져오기
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # h3 요소 가져오기 (콘텐츠가 없는 페이지인 경우 탈출 조건)
        h3_element = soup.find('h3')
        if not h3_element:  # h3 요소가 없다면 더 이상 페이지가 없는 것으로 판단
            print(f"페이지 끝에 도달: 콘텐츠 ID {content_id} - 페이지 {page_num}")
            break
        
        h3_text = h3_element.text.strip()

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
df.to_csv('tour_site_data.csv', index=False, encoding='utf-8-sig')

# 드라이버 종료
driver.quit()

print("데이터 수집 완료! 'tour_site_data.csv' 파일로 저장되었습니다.")
