import requests
from bs4 import BeautifulSoup
import pandas as pd  # pandas 라이브러리 추가

# 웹 페이지 요청
url = "https://www.jeju.go.kr/sobi/kind/kind.htm"
response = requests.get(url)

# BeautifulSoup 객체 생성
soup = BeautifulSoup(response.content, 'html.parser')

# 상점 정보를 저장할 리스트
shops = []

# shop-content 클래스가 포함된 div 요소 찾기
shop_contents = soup.find_all('div', class_='shop-content')

for shop in shop_contents:
    # 상점 제목 추출
    shop_title = shop.find('div', class_='shop-title').text.strip()
    
    # 상점 정보 추출
    addr_info = shop.find('li', class_='shop-info shop-info-addr')
    tel_info = shop.find('li', class_='shop-info shop-info-tel')
    item_info = shop.find('li', class_='shop-info shop-info-item')

    # 정보를 저장할 딕셔너리
    shop_info = {}
    shop_info['상점 제목'] = shop_title
    
    # 주소, 연락처, 품목 정보가 존재할 경우 추가
    if addr_info:
        addr_label = addr_info.find('span', class_='shop-info-label').text.strip()
        addr_text = addr_info.find('span', class_='shop-info-text').text.strip()
        shop_info[addr_label] = addr_text
    
    if tel_info:
        tel_label = tel_info.find('span', class_='shop-info-label').text.strip()
        tel_text = tel_info.find('span', class_='shop-info-text').text.strip()
        shop_info[tel_label] = tel_text
    
    if item_info:
        item_label = item_info.find('span', class_='shop-info-label').text.strip()
        item_text = item_info.find('span', class_='shop-info-text').text.strip().replace('\r\n', ', ')
        shop_info[item_label] = item_text
    
    shops.append(shop_info)

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(shops)

# 데이터프레임을 CSV 파일로 저장
df.to_csv('good_price.csv', index=False, encoding='utf-8-sig')
print("데이터 수집 완료! 'good_price.csv' 파일로 저장되었습니다.")
