import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from datetime import datetime
import chromedriver_autoinstaller

class JejuScraper:
    def __init__(self):
        # Chrome 드라이버 설정
        chromedriver_autoinstaller.install()
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.service = Service()
        
    def get_driver(self):
        return webdriver.Chrome(service=self.service, options=self.options)
    
    def get_content_ids(self, menu_id):
        driver = self.get_driver()
        content_ids = []
        page = 1
        
        try:
            while True:
                url = f"https://visitjeju.net/kr/detail/list?menuId={menu_id}&pageSize=24&page={page}"
                driver.get(url)
                time.sleep(2)  # 페이지 로딩 대기
                
                # Wait for content to load
                wait = WebDriverWait(driver, 10)
                elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a[href*='contentsid=']")))
                
                if not elements:
                    break
                
                # Extract content IDs from URLs
                page_content_ids = []
                for element in elements:
                    href = element.get_attribute('href')
                    if href and 'contentsid=' in href:
                        content_id = href.split('contentsid=')[1].split('&')[0]
                        page_content_ids.append((content_id, menu_id))
                
                if not page_content_ids:
                    break
                    
                content_ids.extend(page_content_ids)
                print(f"Collected {len(content_ids)} items from menu {menu_id}")
                page += 1
                
        except Exception as e:
            print(f"Error collecting content IDs: {e}")
        finally:
            driver.quit()
            
        return content_ids
    
    def scrape_detail(self, content_id, menu_id):
        driver = self.get_driver()
        try:
            url = f"https://visitjeju.net/kr/detail/view?contentsid={content_id}&menuId={menu_id}"
            driver.get(url)
            
            # Wait for main content to load
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h3')))
            
            time.sleep(2)  # Additional wait for dynamic content
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Extract information
            result = {
                '지역명': soup.find('h3').text.strip() if soup.find('h3') else None,
                '주소': soup.find('span', class_='info_sub_tit').text.strip() if soup.find('span', class_='info_sub_tit') else None,
                '태그': ", ".join([tag.text.strip() for tag in soup.find_all('span', class_='tag_txt')]) if soup.find_all('span', class_='tag_txt') else None,
                '리뷰': " | ".join([review.text.strip() for review in soup.find_all('div', class_='review_txt')[:5]]),
                'URL': url,
                'menu_id': menu_id,
                '수집시간': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            print(f"Error scraping {content_id}: {e}")
            return None
        finally:
            driver.quit()
    
    def run(self):
        # Define menu IDs
        menu_ids = [
            "DOM_000001718000000000",  # 관광지
            "DOM_000001719000000000",  # 음식
            "DOM_000001707000000000",  # 숙박
            "DOM_000001720000000000",  # 쇼핑
        ]
        
        all_data = []
        
        # Collect content IDs for each menu
        for menu_id in menu_ids:
            print(f"\nCollecting content IDs for menu {menu_id}")
            content_ids = self.get_content_ids(menu_id)
            
            # Scrape details for each content ID
            print(f"\nScraping details for {len(content_ids)} items from menu {menu_id}")
            for content_id, menu_id in content_ids:
                result = self.scrape_detail(content_id, menu_id)
                if result:
                    all_data.append(result)
                    # Save intermediate results
                    if len(all_data) % 10 == 0:
                        pd.DataFrame(all_data).to_csv('jeju_data_intermediate.csv', index=False, encoding='utf-8-sig')
                        print(f"Saved {len(all_data)} items")
        
        # Save final results
        df = pd.DataFrame(all_data)
        df.to_csv('jeju_data_final.csv', index=False, encoding='utf-8-sig')
        print(f"\nScraping completed. Total items collected: {len(all_data)}")

if __name__ == "__main__":
    scraper = JejuScraper()
    scraper.run()