import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

def scrape_visitjeju_page(content_id, menu_id):
    url = f"https://visitjeju.net/kr/detail/view?contentsid={content_id}&menuId={menu_id}"
    
    # Selenium을 사용하여 페이지 로드
    options = Options()
    options.add_argument('--headless')  # 헤드리스 모드
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    driver.get(url)
    time.sleep(2)  # 페이지 로드 대기
    
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()  # 브라우저 종료

    data = {
        '제목': soup.find('h3').text.strip(),
        '태그': ", ".join([tag.text.strip() for tag in soup.find('div', class_='tag_area').find_all('span')]),
        '주소': soup.find('span', class_='info_sub_cont').text.strip(),
        '조회수': soup.find('span', {'data-v-51160e04': True}).text.strip(),
        '리뷰수': soup.find_all('span', {'data-v-51160e04': True})[1].text.strip(),
        '좋아요 수': soup.find_all('span', {'data-v-51160e04': True})[2].text.strip(),
        '리뷰': [review.text.strip() for review in soup.find_all('div', class_='review_list')]
    }

    return data

def collect_data():
    menu_ids = [
        "DOM_000001718000000000",  # 관광지
        "DOM_000001719000000000",  # 음식
        "DOM_000001707000000000",  # 숙박
        "DOM_000001720000000000",  # 쇼핑
    ]

    data = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for menu_id in menu_ids:
            content_ids = collect_content_ids(menu_id)
            for content_id in content_ids:
                data.append(scrape_visitjeju_page(content_id, menu_id))

    return data

def collect_content_ids(menu_id):
    content_ids = set()
    url = f"https://visitjeju.net/kr/detail/list?menuId={menu_id}&pageSize=12"
    page_num = 1

    # Selenium을 사용하여 Chrome 브라우저 실행
    options = Options()
    options.add_argument('--headless')  # 헤드리스 모드
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    while True:
        driver.get(f"{url}&page={page_num}")
        time.sleep(2)  # 페이지 로드 대기
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        ids_on_page = [link['href'].split("contentsid=")[-1].split("&")[0]
                       for link in soup.select('a[href*="contentsid="]')]

        if not ids_on_page:
            break

        content_ids.update(ids_on_page)
        print(f"Collected {len(ids_on_page)} IDs on page {page_num} for menu {menu_id}")
        page_num += 1

    driver.quit()  # 브라우저 종료
    return content_ids

if __name__ == "__main__":
    data = collect_data()
    for item in data:
        print(item)
