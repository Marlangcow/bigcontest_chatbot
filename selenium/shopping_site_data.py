import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 드라이버 설정
options = Options()
options.add_argument('--headless')
options.add_argument('--blink-settings=imagesEnabled=false')
options.add_argument("window-size=1280x720")
options.page_load_strategy = 'eager'

driver = webdriver.Chrome(options=options)

# URL 및 쇼핑 ID 설정
base_url = "https://visitjeju.net/kr/detail/view?contentsid="
menu_id = "DOM_000001718000000000"

# CSV 파일에서 contents_id 읽기
content_ids_df = pd.read_csv('shopping_site_id.csv').drop_duplicates(subset=["Content ID"])
content_ids = content_ids_df['Content ID'].tolist()

# 데이터를 저장할 리스트 및 배치 크기 설정
data = []
batch_size = 10  # 각 10개 단위로 임시 저장

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
        
        h3_element = soup.find('h3')
        if not h3_element:
            print(f"콘텐츠 ID {content_id}에서 유효한 데이터를 찾을 수 없습니다.")
            continue

        h3_text = h3_element.text.strip()
        tag_area_element = soup.find(class_='tag_area')
        best_tags = [a.text.strip() for a in tag_area_element.find_all('a')] if tag_area_element else []
        p_elements = tag_area_element.find_all('p') if tag_area_element else []
        p_texts = [p.text.strip() for p in p_elements if p.text.strip()]
        
        info_elements = soup.find_all(class_='info_sub_cont')
        info_dict = {info.find_previous_sibling('p').text.strip(): info.text.strip() for info in info_elements}

        print({
            "Title": h3_text,
            "Best Tags": ', '.join(best_tags),
            "Additional P Texts": ', '.join(p_texts),
            **info_dict
        })

        data.append({
            "Title": h3_text,
            "Best Tags": ', '.join(best_tags),
            "Additional P Texts": ', '.join(p_texts),
            **info_dict
        })
        
        # batch_size 단위로 임시 저장
        if idx % batch_size == 0:
            temp_df = pd.DataFrame(data)
            temp_df.to_csv('shopping_site_data_partial.csv', index=False, encoding='utf-8-sig')
            print(f"{idx}개 데이터까지 임시 저장 완료: 'shopping_site_data_partial.csv'")
            
    except:
        print(f"콘텐츠 ID {content_id}에서 데이터를 가져오지 못했습니다.")

# 모든 콘텐츠 ID 수집 완료 후 최종 저장
df = pd.DataFrame(data)
df.to_csv('shopping_site_data.csv', index=False, encoding='utf-8-sig')
driver.quit()
print("데이터 수집 완료! 'shopping_site_data.csv' 파일로 저장되었습니다.")
