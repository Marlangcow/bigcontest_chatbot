import os
import glob
import pandas as pd
import json

# 파일 불러오기
FILE_PATH = "/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA"

# CSV 파일 목록 가져오기
csv_files = glob.glob(os.path.join(FILE_PATH, "*.csv"))

# 데이터프레임을 저장할 딕셔너리
dataframes = {}

# 각 CSV 파일을 읽어 데이터프레임으로 저장
for file in csv_files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    dataframes[file_name] = df

# 맛집
jeju_mct = dataframes['JEJU_MCT_GEO']
jeju_goodprice = dataframes['JEJU_GOODPRICE']
jeju_menus = dataframes['JEJU_MCT_GEO_MENUS']
#요일별/월별
jeju_month = dataframes['JEJU_MONTH_CONCAT']
jeju_weekday = dataframes['JEJU_WKDAY_CONCAT']
# 감정(형태소)
jeju_sentiment = dataframes['JEJU_SENTIMENT_GEO']
jeju_mop = dataframes['JEJU_MOP_GEO']
# 리뷰(크롤링)
jeju_mct_reviews = dataframes['jeju_mct_geo_reviews']
jeju_mop_reviews = dataframes['jeju_mop_geo_reviews']
jeju_sentiment_geo_reviews = dataframes['jeju_sentiment_geo_reviews']


# 각 데이터프레임의 컬럼 정보 출력
for name, df in dataframes.items():
    # globals()를 사용하여 변수명 가져오기
    print(f"\n{name} 컬럼:")
    print(df.columns.tolist())
    print(f"컬럼 수: {len(df.columns)}")
    print("-" * 50)
    
jeju_mct[jeju_mct['detail_address_x'].notna()]

import pandas as pd

# 변경 전
print("변경 전 컬럼명: ", jeju_mct.columns.tolist())
print("변경 전 컬럼명: ", jeju_goodprice.columns.tolist())
print("변경 전 컬럼명: ", jeju_menus.columns.tolist())

# 컬럼명 통일
jeju_mct.drop(columns=['YM', 'OP_YMD', 'ADDR', 'detail_address_x'], inplace=True)
jeju_mct.rename(columns={
    'MCT_NM': '가게명',
    'MCT_TYPE': '업종',
    'basic_address': '주소',
    'UE_CNT_GRP': '이용건수구간',
    'UE_AMT_GRP': '이용금액구간',
    'UE_AMT_PER_TRSN_GRP': '건당평균이용금액구간',
    'MON_UE_CNT_RAT': '월요일이용건수비중',
    'TUE_UE_CNT_RAT': '화요일이용건수비중',
    'WED_UE_CNT_RAT': '수요일이용건수비중',
    'THU_UE_CNT_RAT': '목요일이용건수비중',
    'FRI_UE_CNT_RAT': '금요일이용건수비중',
    'SAT_UE_CNT_RAT': '토요일이용건수비중',
    'SUN_UE_CNT_RAT': '일요일이용건수비중',
    'HR_5_11_UE_CNT_RAT': '5시11시이용건수비중',
    'HR_12_13_UE_CNT_RAT': '12시13시이용건수비중',
    'HR_14_17_UE_CNT_RAT': '14시17시이용건수비중',
    'HR_18_22_UE_CNT_RAT': '18시22시이용건수비중',
    'HR_23_4_UE_CNT_RAT': '23시4시이용건수비중',
    'LOCAL_UE_CNT_RAT': '현지인이용건수비중',
    'RC_M12_MAL_CUS_CNT_RAT': '최근12개월남성회원수비중',
    'RC_M12_FME_CUS_CNT_RAT': '최근12개월여성회원수비중',
    'RC_M12_AGE_UND_20_CUS_CNT_RAT': '최근12개월20대이하회원수비중',
    'RC_M12_AGE_30_CUS_CNT_RAT': '최근12개월30대회원수비중',
    'RC_M12_AGE_40_CUS_CNT_RAT': '최근12개월40대회원수비중',
    'RC_M12_AGE_50_CUS_CNT_RAT': '최근12개월50대회원수비중',
    'RC_M12_AGE_OVR_60_CUS_CNT_RAT': '최근12개월60대이상회원수비중',
    'Latitude': '위도',
    'Longitude': '경도'
}, inplace=True)

jeju_goodprice.drop(columns=['Unnamed: 0', '업소명', '데이터기준일자'], inplace=True)
jeju_goodprice.rename(columns={
    'MCT_NM': '가게명',
    '품목': '메뉴',
    'price': '가격'
}, inplace=True)

jeju_menus.rename(columns={
    'place': '가게명',
    'menu': '메뉴',
    'price': '가격'
}, inplace=True)

# '가게명' 기준으로 합치기
merged_mct = pd.merge(jeju_mct, jeju_goodprice, on='가게명', how='left', suffixes=('', '_goodprice'))
merged_mct = pd.merge(merged_mct, jeju_menus, on='가게명', how='left', suffixes=('', '_menus'))

# 중복된 열 제거 (가장 첫 번째 데이터프레임의 값을 유지)
merged_mct = merged_mct.loc[:, ~merged_mct.columns.duplicated()]

merged_mct.drop(columns=['업종_goodprice', '주소_goodprice', '메뉴_menus'], inplace=True)
merged_mct.columns.tolist()

merged_mct.to_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_MCT.csv', index=False, encoding='utf-8-sig')

import pandas as pd
import json

chunksize = 100000  # 청크 크기 설정
json_merged_result = 'final_mct_json.json'

# 모든 JSON 레코드를 먼저 리스트에 저장합니다.
all_records = []

# CSV 파일을 청크 단위로 읽기
for chunk in pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_MCT.csv', chunksize=chunksize):
    chunk_records = chunk.to_dict(orient='records')
    all_records.extend(chunk_records)

# JSON 파일로 내보내기
with open(json_merged_result, 'w', encoding='utf-8') as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)

# JSON 읽기
merged_result = pd.read_json('/content/final_mct_json.json', encoding='utf-8')

merged_result

# 컬럼 정리

import pandas as pd

# 변경 전
print("변경 전 컬럼명: ", jeju_month.columns.tolist())
print("변경 전 컬럼명: ", jeju_weekday.columns.tolist())
print("변경 전 컬럼명: ", jeju_sentiment.columns.tolist())
print("변경 전 컬럼명: ", jeju_mop.columns.tolist())

# 컬럼명 통일
# 변경 전 컬럼명:  ['CL_CD', 'CL_NM', 'AREA_NM', 'ADDR', 'BASE_YEAR', 'ALL_TOTAL_CO', 'JAN_VIEWS_CO', 'FEB_VIEWS_CO', 'MAR_VIEWS_CO', 'APR_VIEWS_CO', 'MAY_VIEWS_CO', 'JUN_VIEWS_CO', 'JULY_VIEWS_CO', 'AUG_VIEWS_CO', 'SEP_VIEWS_CO', 'OCT_VIEWS_CO', 'NOV_VIEWS_CO', 'DEC_VIEWS_CO']
jeju_month.drop(columns=['CL_CD','BASE_YEAR'], inplace=True)
jeju_month.rename(columns={
  'AREA_NM': '관광지명',
  'CL_NM': '업종',
  'ADDR': '주소',
  'ALL_TOTAL_CO': '전체총합수',
  'JAN_VIEWS_CO': '1월조회수',
  'FEB_VIEWS_CO': '2월조회수',
  'MAR_VIEWS_CO': '3월조회수',
  'APR_VIEWS_CO': '4월조회수',
  'MAY_VIEWS_CO': '5월조회수',
  'JUN_VIEWS_CO': '6월조회수',
  'JULY_VIEWS_CO': '7월조회수',
  'AUG_VIEWS_CO': '8월조회수',
  'SEP_VIEWS_CO': '9월조회수',
  'OCT_VIEWS_CO': '10월조회수',
  'NOV_VIEWS_CO': '11월조회수',
  'DEC_VIEWS_CO': '12월조회수'
}, inplace=True)

# 변경 전 컬럼명:  ['CL_CD', 'CL_NM', 'AREA_NM', 'ADDR', 'BASE_YEAR', 'BASE_MT', 'ALL_TOTAL_CO', 'MON_VIEWS_CO', 'TUES_VIEWS_CO', 'WED_VIEWS_CO', 'THUR_VIEWS_CO', 'FRI_VIEWS_CO', 'SAT_VIEWS_CO', 'SUN_VIEWS_CO']
jeju_weekday.drop(columns=['CL_CD', 'BASE_YEAR'], inplace=True)
jeju_weekday.rename(columns={
  'AREA_NM': '관광지명',
  'CL_NM': '업종',
  'ADDR': '주소',
  'BASE_MT': '기준월',
  'ALL_TOTAL_CO': '전체총합수',
  'MON_VIEWS_CO': '월요일조회수',
  'TUES_VIEWS_CO': '화요일조회수',
  'WED_VIEWS_CO': '수요일조회수',
  'THUR_VIEWS_CO': '목요일조회수',
  'FRI_VIEWS_CO': '금요일조회수',
  'SAT_VIEWS_CO': '토요일조회수',
  'SUN_VIEWS_CO': '일요일조회수'
}, inplace=True)

# 변경 전 컬럼명:  ['CL_NM', 'TRRSRT_NM', 'TRRSRT_ADDR', 'ORIGIN_CL_NM', 'ANALS_BEGIN_DE', 'ANALS_END_DE', 'AVRG_SCORE_VALUE', 'CORE_KWRD_CN', 'CORE_KWRD_CO', 'AFRM_KWRD_CN', 'AFRM_KWRD_CO', 'NEGA_KWRD_CN', 'NEGA_KWRD_CO']
jeju_sentiment.drop(columns=['ANALS_BEGIN_DE','ANALS_END_DE'], inplace=True)
jeju_sentiment.rename(columns={
  'CL_NM': '업종',
  'TRRSRT_NM': '관광지명',
  'TRRSRT_ADDR': '주소',
  'ORIGIN_CL_NM': '출처',
  'AVRG_SCORE_VALUE': '평균평점',
  'CORE_KWRD_CN': '핵심키워드',
  'CORE_KWRD_CO': '핵심키워드수',
  'AFRM_KWRD_CN': '긍정키워드',
  'AFRM_KWRD_CO': '긍정키워드수',
  'NEGA_KWRD_CN': '부정키워드',
  'NEGA_KWRD_CO': '부정키워드수'
}, inplace=True)

# 변경 전 컬럼명:  ['BASE_YM', 'CL_NM', 'TRRSRT_NM', 'TRRSRT_ADDR', 'SCORE_VALUE', 'MOP_CN', 'REGIST_DE']
jeju_mop.drop(columns=['BASE_YM', 'REGIST_DE' ], inplace=True)
jeju_mop.rename(columns={
  'TRRSRT_NM': '관광지명',
  'CL_NM': '업종',
  'TRRSRT_ADDR': '주소',
  'SCORE_VALUE': '평균평점',
  'MOP_CN': '키워드'
}, inplace=True)

# 변경 후
print("변경 후 컬럼명: ", jeju_month.columns.tolist())
print("변경 후 컬럼명: ", jeju_weekday.columns.tolist())
print("변경 후 컬럼명: ", jeju_sentiment.columns.tolist())
print("변경 후 컬럼명: ", jeju_mop.columns.tolist())



import pandas as pd

# 데이터프레임의 컬럼명 통일

# '관광지명' 기준으로 합치기
# 월별+요일별
merged_trrsrt = pd.merge(jeju_month, jeju_weekday, on='관광지명', how='left', suffixes=('', '_weekday'))
# 평점+형태소
merged_mop_sentiment = pd.merge(jeju_mop, jeju_sentiment, on='관광지명', how='left', suffixes=('', '_sentiment'))

# # 중복된 열 제거 (가장 첫 번째 데이터프레임의 값을 유지)
merged_trrsrt = merged_trrsrt.loc[:, ~merged_trrsrt.columns.duplicated()]
merged_mop_sentiment = merged_mop_sentiment.loc[:, ~merged_mop_sentiment.columns.duplicated()]

# 결과 출력
print(merged_trrsrt.columns.tolist())
print(merged_mop_sentiment.columns.tolist())

# 중복되는 컬럼 제거
merged_trrsrt.drop(columns=['업종_weekday', '주소_weekday', '전체총합수_weekday'], inplace=True)
merged_mop_sentiment.drop(columns=['업종_sentiment', '주소_sentiment', '평균평점_sentiment'], inplace=True)

# 데이터 유형 변경
merged_trrsrt = merged_trrsrt.astype({'전체총합수': 'int32', '1월조회수': 'float32', '2월조회수': 'float32', '기준월': 'int32'})
merged_mop_sentiment = merged_mop_sentiment.astype({'평균평점': 'float32', '핵심키워드수': 'float32', '긍정키워드수': 'float32', '부정키워드수': 'float32'})

# 병합
merged_result = merged_trrsrt.merge(merged_mop_sentiment, on=['관광지명', '업종'], how='outer')

merged_result.drop(columns=['주소_y'], inplace=True)
merged_result.rename(columns={'주소_x': '주소'}, inplace=True)

# csv로 내보내기
merged_result.to_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_TRRSRT.csv', index=False, encoding='utf-8-sig')

import pandas as pd
import json

chunksize = 100000  # 청크 크기 설정
# 파일 이름을 포함한 JSON 파일 경로
json_trrsrt_result = '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/final_trrsrt.json'

# 모든 JSON 레코드를 먼저 리스트에 저장합니다.
all_records = []

# CSV 파일을 청크 단위로 읽기
for chunk in pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_TRRSRT.csv', chunksize=chunksize):
    chunk_records = chunk.to_dict(orient='records')
    all_records.extend(chunk_records)

# JSON 파일로 내보내기
with open(json_trrsrt_result, 'w', encoding='utf-8') as f:
    json.dump(all_records, f, ensure_ascii=False, indent=2)

# DataFrame을 JSON 파일로 저장하기
json_merged_result.to_json('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_TRRSRT.json')

import pandas as pd

# 데이터 불러오기
mct_review = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA/jeju_mct_geo_reviews.csv')
mop_review= pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA/jeju_mop_geo_reviews.csv')
sentiment_review = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA/jeju_sentiment_geo_reviews.csv')

# 행렬변환
mct_review = mct_review.T
mop_review = mop_review.T
sentiment_review = sentiment_review.T

# 컬럼명 변경
mct_review.columns = mct_review.columns.astype(str)  # 컬럼명을 문자열로 변환
mct_review.insert(0, '관광지명', mct_review.index)
mct_review.reset_index(drop=True, inplace=True)
mct_review.rename(columns={
    '0': 'review_1',
    '1': 'review_2',
    '2': 'review_3',
    '3': 'review_4',
    '4': 'review_5'
}, inplace=True)

mop_review.columns = mop_review.columns.astype(str)  # 컬럼명을 문자열로 변환
mop_review.insert(0, '관광지명', mop_review.index)
mop_review.reset_index(drop=True, inplace=True)
mop_review.rename(columns={
    '0': 'review_1',
    '1': 'review_2',
    '2': 'review_3',
    '3': 'review_4',
    '4': 'review_5'
}, inplace=True)

sentiment_review.columns = sentiment_review.columns.astype(str)  # 컬럼명을 문자열로 변환
sentiment_review.insert(0, '관광지명', sentiment_review.index)
sentiment_review.reset_index(drop=True, inplace=True)
sentiment_review.rename(columns={
    '0': 'review_1',
    '1': 'review_2',
    '2': 'review_3',
    '3': 'review_4',
    '4': 'review_5'
}, inplace=True)

# 변환 함수 정의
import ast

def extract_contents_and_point(review):
    try:
        # 문자열을 딕셔너리로 변환
        review_dict = ast.literal_eval(review)
        # 'contents'와 'point' 값 추출
        contents = review_dict.get('contents', '')
        point = review_dict.get('point', '')
        # 'contents'와 'point'를 하나의 문자열로 결합
        return f"{contents} (평점: {point})"
    except (ValueError, SyntaxError):
        # 변환에 실패한 경우 원본 값을 그대로 반환
        return review

# review_1부터 review_5까지 모든 컬럼에 대해 변환 적용
for col in ['review_1', 'review_2', 'review_3', 'review_4', 'review_5']:
    mct_review[col] = mct_review[col].apply(extract_contents_and_point)
    mop_review[col] = mct_review[col].apply(extract_contents_and_point)
    sentiment_review[col] = mct_review[col].apply(extract_contents_and_point)
    
# 데이터프레임을 세로로 합치기
combined_review = pd.concat([mct_review, mop_review, sentiment_review], axis=0, ignore_index=True)

# 결과 확인
print(combined_review)

# CSV로 내보내기
combined_review.to_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_REVIEW', index=True)

# JSON으로 내보내기
combined_review.to_json('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/final_review.json', orient='records', lines=True)

# 결과 확인 (옵션)
print("JSON 파일이 성공적으로 저장되었습니다.")

# JSON 파일 불러오기
merged_reviews = pd.read_json('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/final_review.json', lines=True)

# 결과 확인
print(merged_reviews.head())  # DataFrame의 처음 5행 출력

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import dask.dataframe as dd

# Dask 데이터프레임으로 CSV 로드
mct_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_MCT.csv',
      dtype={
        '가게명': 'string',
        '업종': 'string',
        '연락처': 'string',
        '영업정보': 'string',
        '지역': 'string',
        '메뉴': 'string',
        '가격': 'string',
        '주소': 'string',
        # 여기에 필요한 다른 열들에 대한 dtype 추가
    }
)
trrsrt_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_TRRSRT.csv',
      dtype={
        '업종': 'string',
        '관광지명': 'string',
        '주소': 'string',
        '전체총합수': 'float64',
        '1월조회수': 'float64',
        '2월조회수': 'float64',
        '3월조회수': 'float64',
        '4월조회수': 'float64',
        '5월조회수': 'float64',
        '6월조회수': 'float64',
        '7월조회수': 'float64',
        '8월조회수': 'float64',
        '9월조회수': 'float64',
        '10월조회수': 'float64',
        '11월조회수': 'float64',
        '12월조회수': 'float64',
        '기준월': 'string',
        '월요일조회수': 'float64',
        '화요일조회수': 'float64',
        '수요일조회수': 'float64',
        '목요일조회수': 'float64',
        '금요일조회수': 'float64',
        '토요일조회수': 'float64',
        '일요일조회수': 'float64',
        '평균평점': 'float64',
        '키워드': 'string',  # 변경
        '출처': 'string',    # 변경
        '핵심키워드': 'string',  # 변경
        '핵심키워드수': 'float64',
        '긍정키워드': 'string',  # 변경
        '긍정키워드수': 'float64',
        '부정키워드': 'string',  # 변경
        '부정키워드수': 'float64',
    }
)
review_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_REVIEW.csv',
      dtype={
        '관광지명': 'string',
        'review_1': 'string',
        'review_2': 'string',
        'review_3': 'string',
        'review_4': 'string',
        'review_5': 'string',
    }
)

import pandas as pd

# 문자열을 카테고리로 변환
string_columns = ['가게명', '업종', '주소', '지역', '연락처', '영업정보', '메뉴', '가격']
for col in string_columns:
    mct_ddf[col] = mct_ddf[col].astype('category')

# 정수형 타입을 최적화
int_columns = ['이용건수구간', '이용금액구간', '건당평균이용금액구간']
for col in int_columns:
    mct_ddf[col] = mct_ddf[col].astype('int32')  # int16이 가능하다면 더 줄일 수 있음

# 부동소수점 타입을 float32로 변환
float_columns = [
    '월요일이용건수비중', '화요일이용건수비중', '수요일이용건수비중',
    '목요일이용건수비중', '금요일이용건수비중', '토요일이용건수비중',
    '일요일이용건수비중', '5시11시이용건수비중', '12시13시이용건수비중',
    '14시17시이용건수비중', '18시22시이용건수비중', '23시4시이용건수비중',
    '현지인이용건수비중', '최근12개월남성회원수비중', '최근12개월여성회원수비중',
    '최근12개월20대이하회원수비중', '최근12개월30대회원수비중',
    '최근12개월40대회원수비중', '최근12개월50대회원수비중',
    '최근12개월60대이상회원수비중', '위도', '경도'
]
for col in float_columns:
    mct_ddf[col] = mct_ddf[col].astype('float32')

# 최적화된 데이터프레임 정보 출력
print(mct_ddf.info())

# 결과를 Pandas 데이터프레임으로 변환
mct_df = mct_ddf.compute()

# 결과를 CSV로 저장
mct_df.to_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/CLEANED_MCT.csv',
            index=False, encoding='utf-8-sig')  # 인덱스 없이 저장하고 UTF-8 BOM 형식으로 인코딩

print("CSV 파일로 저장 완료: CLEANED_MCT.csv")

import pandas as pd

# 데이터프레임 생성 (예시)

# 문자열을 카테고리로 변환
string_columns = ['업종', '관광지명', '주소', '기준월', '키워드', '출처', '핵심키워드', '긍정키워드', '부정키워드']
for col in string_columns:
    trrsrt_df[col] = trrsrt_df[col].astype('category')

# 부동소수점 타입을 float32로 변환
float_columns = ['전체총합수', '평균평점',
                 '1월조회수', '2월조회수', '3월조회수', '4월조회수',
                 '5월조회수', '6월조회수', '7월조회수', '8월조회수',
                 '9월조회수', '10월조회수', '11월조회수', '12월조회수',
                 '월요일조회수', '화요일조회수', '수요일조회수',
                 '목요일조회수', '금요일조회수', '토요일조회수', '일요일조회수']
for col in float_columns:
    trrsrt_df[col] = trrsrt_df[col].astype('float32')

# 핵심키워드수, 긍정키워드수, 부정키워드수를 처리하기
int_columns = ['핵심키워드수', '긍정키워드수', '부정키워드수']

# NaN 값을 0으로 대체한 후 int32로 변환
for col in int_columns:
    trrsrt_df[col] = trrsrt_df[col].fillna(0).astype('int32')

# 최적화된 데이터프레임 정보 출력
print(trrsrt_df.info())

# 결과를 Pandas 데이터프레임으로 변환
trrsrt_df = trrsrt_ddf.compute()

# 결과를 CSV로 저장
trrsrt_df.to_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/CLEANED_TRRSRT.csv',
            index=False, encoding='utf-8-sig')  # 인덱스 없이 저장하고 UTF-8 BOM 형식으로 인코딩

print("CSV 파일로 저장 완료: CLEANED_TRRSRT.csv")

# Dask 데이터프레임 로드 및 전처리 코드 (이전 코드)
import dask.dataframe as dd

# Dask 데이터프레임 로드
review_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_REVIEW',
      dtype={
          '관광지명': 'string',
          'review_1': 'string',
          'review_2': 'string',
          'review_3': 'string',
          'review_4': 'string',
          'review_5': 'string',
      }
)

# Unnamed: 0 컬럼 제거
review_ddf = review_ddf.drop(columns=['Unnamed: 0'])

# review_1부터 review_5까지 값이 모두 비어있는 행 삭제
review_ddf = review_ddf.dropna(subset=['review_1', 'review_2', 'review_3', 'review_4', 'review_5'], how='all')

# 결과를 Pandas 데이터프레임으로 변환
review_df = review_ddf.compute()

# 결과를 CSV로 저장
review_df.to_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/CLEANED_REVIEW.csv',
            index=False, encoding='utf-8-sig')  # 인덱스 없이 저장하고 UTF-8 BOM 형식으로 인코딩

print("CSV 파일로 저장 완료: CLEANED_REVIEW.csv")

import dask.dataframe as dd
review_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/CLEANED_REVIEW.csv')
review_ddf.head()

# 프로그래스 바 처리
!pip install tqdm
from tqdm import tqdm

tqdm.pandas()

import dask.dataframe as dd

# Dask 데이터프레임으로 CSV 로드
mct_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_MCT.csv',
      dtype={
        '가게명': 'string',
        '업종': 'string',
        '연락처': 'string',
        '영업정보': 'string',
        '지역': 'string',
        '메뉴': 'string',
        '가격': 'string',
        '주소': 'string',
        # 여기에 필요한 다른 열들에 대한 dtype 추가
    }
)
trrsrt_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_TRRSRT.csv',
      dtype={
        '업종': 'string',
        '관광지명': 'string',
        '주소': 'string',
        '전체총합수': 'float64',
        '1월조회수': 'float64',
        '2월조회수': 'float64',
        '3월조회수': 'float64',
        '4월조회수': 'float64',
        '5월조회수': 'float64',
        '6월조회수': 'float64',
        '7월조회수': 'float64',
        '8월조회수': 'float64',
        '9월조회수': 'float64',
        '10월조회수': 'float64',
        '11월조회수': 'float64',
        '12월조회수': 'float64',
        '기준월': 'string',
        '월요일조회수': 'float64',
        '화요일조회수': 'float64',
        '수요일조회수': 'float64',
        '목요일조회수': 'float64',
        '금요일조회수': 'float64',
        '토요일조회수': 'float64',
        '일요일조회수': 'float64',
        '평균평점': 'float64',
        '키워드': 'string',
        '출처': 'string',
        '핵심키워드': 'string',
        '핵심키워드수': 'float64',
        '긍정키워드': 'string',
        '긍정키워드수': 'float64',
        '부정키워드': 'string',
        '부정키워드수': 'float64',
    }
)
review_ddf = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/FINAL_REVIEW.csv',
      dtype={
        '관광지명': 'string',
        'review_1': 'string',
        'review_2': 'string',
        'review_3': 'string',
        'review_4': 'string',
        'review_5': 'string',
    }
)
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

def create_documents(df, info_type):
    # 모든 컬럼을 "컬럼명: 값" 형식으로 변환
    document_texts = df.apply(lambda row: ', '.join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]), axis=1)

    # 리뷰 데이터가 있는 경우 추가
    if info_type == "리뷰":
        review_columns = [col for col in df.columns if col.startswith('review_')]
        review_texts = df[review_columns].fillna('').apply(lambda row: ' | '.join(filter(None, row)), axis=1)
        document_texts += ", 리뷰: " + review_texts

    # 최종 문서 리스트 생성
    documents = [{"text": f"{info_type}: {text}"} for text in document_texts]
    return documents

def save_documents_to_disk(docs, file_path):
    try:
        pd.DataFrame(docs).to_csv(file_path, mode='a', index=False, header=True, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error saving documents: {e}")

def process_data(ddf, info_type, file_path, chunk_size=20000):
    all_documents = []  # 모든 문서를 수집할 리스트

    # Dask DataFrame을 청크로 나누어 처리
    for chunk in tqdm(ddf.to_delayed(), desc=f"Processing {info_type}", unit="chunk"):
        chunk_df = chunk.compute()  # 청크를 컴퓨팅하여 Pandas DataFrame으로 변환
        documents = create_documents(chunk_df, info_type)  # 문서 생성
        all_documents.extend(documents)  # 문서 리스트에 추가

        # chunk_size 개수마다 문서 저장
        if len(all_documents) >= chunk_size:
            save_documents_to_disk(all_documents, file_path)
            all_documents.clear()  # 리스트 초기화

    # 남아있는 문서 저장
    if all_documents:
        save_documents_to_disk(all_documents, file_path)

process_data(review_ddf, "리뷰", '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/review_documents.csv')
process_data(mct_ddf, "가게 정보", '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/mct_documents.csv')
process_data(trrsrt_ddf, "관광지 정보", '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/trrsrt_documents.csv')

import dask.dataframe as dd
import pandas as pd
import re
from langchain.document_loaders import DataFrameLoader
from langchain.docstore.document import Document
from dask.diagnostics import ProgressBar
from langchain.text_splitter import CharacterTextSplitter

# CSV 파일 로드
review_doc = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/review_documents.csv', header=None)
mct_doc = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/mct_documents.csv', header=None)
trrsrt_doc = dd.read_csv('/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/trrsrt_documents.csv', header=None)

# 메타데이터 추출 함수 정의
def extract_metadata(row, metadata_keys):
    text = row[0] if isinstance(row[0], str) else ''

    # 정규 표현식을 사용하여 필요한 정보 추출
    metadata_dict = {}
    for key in metadata_keys:
        match = re.search(f'{key}: (.+?)(?:,|$)', text)
        metadata_dict[key] = match.group(1).strip() if match else None  # 공백 제거
    return metadata_dict

# TRRSRT 메타데이터 키 정의
trrsrt_metadata_keys = ['관광지명', '업종', '주소']

# Dask DataFrame을 사용한 TRRSRT 메타데이터 추출
with ProgressBar():
    trrsrt_metadata = trrsrt_doc.map_partitions(
        lambda df: df.apply(lambda row: extract_metadata(row, trrsrt_metadata_keys), axis=1)
    ).compute()

# TRRSRT 메타데이터 변환
trrsrt_metadata_df = pd.DataFrame(trrsrt_metadata.tolist(), columns=trrsrt_metadata_keys)

# 문서 로더 설정
trrsrt_loader = DataFrameLoader(trrsrt_metadata_df, page_content_column='관광지명')

# 문서 분할기 설정
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)

# 메타데이터와 ID 부여 함수
def assign_metadata_and_id(docs, metadata, prefix='doc'):
    return [
        Document(page_content=doc['관광지명'], metadata={**meta, 'id': f'{prefix}_{idx}'})
        for idx, (doc, meta) in enumerate(zip(docs, metadata))
    ]

# 문서 로드 및 메타데이터, ID 부여
def load_documents(loader, metadata, prefix):
    docs = loader.load()
    return assign_metadata_and_id(docs, metadata.to_dict(orient='records'), prefix)

# REVIEW, MCT, TRRSRT 문서 로드
review_loader = DataFrameLoader(review_doc, page_content_column=0)
mct_loader = DataFrameLoader(mct_doc, page_content_column=0)

review_docs = load_documents(review_loader, review_metadata, 'review')
mct_docs = load_documents(mct_loader, mct_metadata, 'mct')
trrsrt_docs_with_meta = load_documents(trrsrt_loader, trrsrt_metadata_df, 'trrsrt')

# 문서 내용 및 메타데이터 확인
for doc in review_docs:
    print("리뷰 문서 내용:", doc.page_content)
    print("리뷰 메타데이터:", doc.metadata)

for doc in mct_docs:
    print("가게 문서 내용:", doc.page_content)
    print("가게 메타데이터:", doc.metadata)

for doc in trrsrt_docs_with_meta:
    print("관광지 문서 내용:", doc.page_content)
    print("관광지 메타데이터:", doc.metadata)

# doc들을 Pickle 파일로 저장하는 함수
def save_docs_to_cache(docs, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(docs, f)

# 리뷰, MCT, TRRSRT 문서 저장
save_docs_to_cache(review_docs, '/content/drive/MyDrive/Colab Notebooks/review_docs.pkl')
save_docs_to_cache(mct_docs, '/content/drive/MyDrive/Colab Notebooks/mct_docs.pkl')
save_docs_to_cache(trrsrt_docs, '/content/drive/MyDrive/Colab Notebooks/trrsrt_docs.pkl')

import os
import pickle
import re
import dask.dataframe as dd
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from dask.diagnostics import ProgressBar
import ast

# 캐시 파일 경로 설정
review_metadata_cache = '/content/drive/MyDrive/Colab Notebooks/review_metadata.pkl'
mct_metadata_cache = '/content/drive/MyDrive/Colab Notebooks/mct_metadata.pkl'
trrsrt_metadata_cache = '/content/drive/MyDrive/Colab Notebooks/trrsrt_metadata_final.pkl'

# 문자열에서 메타데이터를 추출하는 함수
def extract_metadata(row, metadata_keys):
    text = row[0] if isinstance(row[0], str) else ''

    # 정규 표현식을 사용하여 필요한 정보 추출
    metadata_dict = {}
    for key in metadata_keys:
        # 정규 표현식 수정
        match = re.search(f'{key}: (.+?)(?:,|$)', text)
        metadata_dict[key] = match.group(1).strip() if match else None  # strip()을 추가하여 공백 제거

    return metadata_dict

# 메타데이터 키 정의
review_metadata_keys = ['관광지명']
mct_metadata_keys = ['가게명', '업종', '주소']
trrsrt_metadata_keys = ['관광지명', '업종', '주소']

# 캐시 로드 함수
def load_cached_metadata(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

# 캐시 저장 함수
def save_metadata_to_cache(metadata, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(metadata, f)

# 각 메타데이터 로드 및 처리
review_metadata = load_cached_metadata(review_metadata_cache)
if review_metadata is None:
    review_metadata = review_doc.apply(lambda row: extract_metadata(row, review_metadata_keys), axis=1, result_type='expand')
    save_metadata_to_cache(review_metadata, review_metadata_cache)

mct_metadata = load_cached_metadata(mct_metadata_cache)
if mct_metadata is None:
    mct_metadata = mct_doc.apply(lambda row: extract_metadata(row, mct_metadata_keys), axis=1, result_type='expand')
    save_metadata_to_cache(mct_metadata, mct_metadata_cache)

trrsrt_metadata = load_cached_metadata(trrsrt_metadata_cache)
if trrsrt_metadata is None:
    trrsrt_metadata = mct_doc.apply(lambda row: extract_metadata(row, trrsrt_metadata_keys), axis=1, result_type='expand')
    save_metadata_to_cache(trrsrt_metadata, trrsrt_metadata_keys)

import os
import pickle
import re

# 캐시 경로 정의
trrsrt_metadata_cache = '/content/drive/MyDrive/Colab Notebooks/trrsrt_metadata_final.pkl'

# 문자열에서 메타데이터를 추출하는 함수
def extract_metadata(row, metadata_keys):
    text = row[0] if isinstance(row[0], str) else ''

    # 정규 표현식을 사용하여 필요한 정보 추출
    metadata_dict = {}
    for key in metadata_keys:
        # 정규 표현식 수정
        match = re.search(f'{key}: (.+?)(?:,|$)', text)
        metadata_dict[key] = match.group(1).strip() if match else None  # strip()을 추가하여 공백 제거

    return metadata_dict

# 메타데이터 키 정의
trrsrt_metadata_keys = ['관광지명', '업종', '주소']

# 캐시 로드 함수
def load_cached_metadata(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

# 캐시 저장 함수
def save_metadata_to_cache(metadata, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(metadata, f)

# 각 메타데이터 로드 및 처리
trrsrt_metadata = load_cached_metadata(trrsrt_metadata_cache)
if trrsrt_metadata is None:
    # 메타데이터가 캐시에 없다면 mct_doc을 사용하여 생성
    trrsrt_metadata = mct_doc.apply(
        lambda row: extract_metadata(row, trrsrt_metadata_keys),
        axis=1,
        result_type='expand'
    )
    # 생성한 메타데이터를 캐시에 저장
    save_metadata_to_cache(trrsrt_metadata, trrsrt_metadata_cache)

import os
os.environ['MECAB_CONFIG'] = '/usr/local/lib/python3.10/dist-packages/konlpy/tag/_mecab.py'

import os
import pickle

# Pickle 파일에서 doc 불러오는 함수
def load_docs_from_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

# doc들 불러오기
review_docs = load_docs_from_cache('/content/drive/MyDrive/Colab Notebooks/review_docs.pkl')
mct_docs = load_docs_from_cache('/content/drive/MyDrive/Colab Notebooks/mct_docs.pkl')
trrsrt_docs = load_docs_from_cache('/content/drive/MyDrive/Colab Notebooks/trrsrt_docs.pkl')

# 문서 확인
print("review_docs 내용:", review_docs[0])
print("review_docs 길이:", len(review_docs))
print("-------")
print("mct_docs 내용:", mct_docs[0])
print("mct_docs 길이:", len(mct_docs))
print("-------")
print("trrsrt_docs 내용:", trrsrt_docs[0])
print("trrsrt_docs 길이:", len(trrsrt_docs))

# 토큰 확인
import json

# JSON 파일을 불러오는 함수
def load_tokens_from_file(filename):
    # try-except 블록을 사용하여 JSONDecodeError 처리
    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"{filename} 파일에서 JSON 디코딩 오류: {e}")
        # 오류를 다르게 처리하거나 빈 리스트를 반환하도록 선택할 수 있습니다.
        return []

# 리뷰, 가게 정보, 관광지 정보 토큰을 파일에서 불러오기
review_tokens = load_tokens_from_file("review_tokens.json")
mct_tokens = load_tokens_from_file("mct_tokens.json")
trrsrt_tokens = load_tokens_from_file("trrsrt_tokens.json")

# 결과 확인
print("리뷰 토큰:", review_tokens[:50])
print("가게 정보 토큰:", mct_tokens[:50])
print("관광지 정보 토큰:", trrsrt_tokens[:50])

import json

# JSON 파일을 불러오는 함수
def load_tokens_from_file(filename):
    # try-except 블록을 사용하여 JSONDecodeError 처리
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"{filename} 파일에서 JSON 디코딩 오류: {e}")
        return []

# 불필요한 토큰 정의
unwanted_tokens = ['(', ')', '##', '은', '는', '이', '가', '사']

def filter_tokens(tokens):
    """불필요한 토큰을 필터링하는 함수"""
    return [token for token in tokens if token not in unwanted_tokens and not token.startswith('##')]

def save_tokens_to_file(tokens, filename):
    """토큰을 JSON 파일로 저장하는 함수"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=4)
    print(f"토큰이 {filename} 파일에 저장되었습니다.")

# 리뷰, 가게 정보, 관광지 정보 토큰을 파일에서 불러오기
review_tokens = load_tokens_from_file("review_tokens.json")
mct_tokens = load_tokens_from_file("mct_tokens.json")
trrsrt_tokens = load_tokens_from_file("trrsrt_tokens.json")

# 토큰 필터링
filtered_review_tokens = filter_tokens(review_tokens)
filtered_mct_tokens = filter_tokens(mct_tokens)
filtered_trrsrt_tokens = filter_tokens(trrsrt_tokens)

# 결과 확인
print("필터링된 리뷰 토큰:", filtered_review_tokens[:50])
print("필터링된 가게 정보 토큰:", filtered_mct_tokens[:50])
print("필터링된 관광지 정보 토큰:", filtered_trrsrt_tokens[:50])

# 필터링된 토큰 저장
save_tokens_to_file(filtered_review_tokens, "filtered_review_tokens.json")
save_tokens_to_file(filtered_mct_tokens, "filtered_mct_tokens.json")
save_tokens_to_file(filtered_trrsrt_tokens, "filtered_trrsrt_tokens.json")

import torch
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 및 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# JSON 파일을 불러오는 함수
def load_tokens_from_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"{filename} 파일에서 JSON 디코딩 오류: {e}")
        return []

# 리뷰, 가게 정보, 관광지 정보 토큰을 파일에서 불러오기
filtered_review_tokens = load_tokens_from_file("filtered_review_tokens.json")
filtered_mct_tokens = load_tokens_from_file("filtered_mct_tokens.json")
filtered_trrsrt_tokens = load_tokens_from_file("filtered_trrsrt_tokens.json")

def stratified_sample_by_ratio(tokens, class_labels, sampling_ratio):
    class_samples = {}
    for token, label in zip(tokens, class_labels):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(token)

    sampled_tokens = []
    for class_label, class_tokens in class_samples.items():
        num_samples = int(len(class_tokens) * sampling_ratio)
        sampled = np.random.choice(class_tokens, size=min(num_samples, len(class_tokens)), replace=False)
        sampled_tokens.extend(sampled)

    return sampled_tokens

def embed_tokens_mixed_precision(token_list, model, tokenizer, batch_size=32):
    if not token_list:
        return np.array([])

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(token_list), batch_size), desc="Embedding Tokens"):
            batch_tokens = token_list[i:i + batch_size]
            inputs = tokenizer(batch_tokens, padding=True, truncation=True, return_tensors="pt").to(device)

            with autocast():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

# 비율 샘플링 비율 설정
sampling_ratio = 0.1

# 클래스 레이블 정의
trrsrt_class_labels = [token if isinstance(token, str) and token in ['관광지', '음식점', '숙박'] else 'default_class' for token in filtered_trrsrt_tokens]
review_class_labels = [token if isinstance(token, str) and token in ['긍정', '부정', '중립'] else 'default_class' for token in filtered_review_tokens]
mct_class_labels = [token if isinstance(token, str) and token in ['식당', '카페', '기타'] else 'default_class' for token in filtered_mct_tokens]

# 층화 샘플링 수행
trrsrt_tokens_sampled = stratified_sample_by_ratio(filtered_trrsrt_tokens, trrsrt_class_labels, sampling_ratio)
review_tokens_sampled = stratified_sample_by_ratio(filtered_review_tokens, review_class_labels, sampling_ratio)
mct_tokens_sampled = stratified_sample_by_ratio(filtered_mct_tokens, mct_class_labels, sampling_ratio)

# 혼합 정밀도를 사용하여 임베딩
trrsrt_embeddings = embed_tokens_mixed_precision(trrsrt_tokens_sampled, model, tokenizer, batch_size=32)
review_embeddings = embed_tokens_mixed_precision(review_tokens_sampled, model, tokenizer, batch_size=32)
mct_embeddings = embed_tokens_mixed_precision(mct_tokens_sampled, model, tokenizer, batch_size=32)

# 임베딩 저장
np.save("trrsrt_embeddings.npy", trrsrt_embeddings.astype(np.float32))
print("TRRSRT 임베딩 저장 완료.")
np.save("review_embeddings.npy", review_embeddings.astype(np.float32))
print("리뷰 임베딩 저장 완료.")
np.save("mct_embeddings.npy", mct_embeddings.astype(np.float32))
print("MCT 임베딩 저장 완료.")

# CSV 로드
csv_file_paths = [
    '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/review_documents.csv',
    '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/mct_documents.csv',
    '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/trrsrt_documents.csv'
]

dfs = [pd.read_csv(csv_file_path) for csv_file_path in csv_file_paths]

# CSV 파일 미리보기
for i, dfs in enumerate(dfs):
  print(f"CSV 파일 {i+1} 데이터 크기:", dfs.shape)
  print(dfs.head())
  print()
  
# 임베딩 로드
npy_file_paths = [
    '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/V2/review_embeddings_v2.npy',
    '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/V2/mct_embeddings_v2.npy',
    '/content/drive/MyDrive/Colab Notebooks/부트캠프_한경 with 토스뱅크/프로젝트/공모전_빅콘테스트/DATA_FINAL/V2/trrsrt_embeddings_v2.npy'
 ]

# embedding = np.load(npy_file_paths)
embeddings_list = [np.load(npy_file_path) for npy_file_path in npy_file_paths]

# 임베딩된 데이터의 크기 확인
for i, embeddings in enumerate(embeddings_list):
    print(f"임베딩 데이터 {i+1} 크기: {embeddings.shape}")

# 임베딩된 데이터의 크기 확인 및 배치로 저장
for i, embeddings in enumerate(embeddings_list):
    print(f"임베딩 데이터 {i+1} 크기: {embeddings.shape}")

    # 각 임베딩을 개별 파일로 저장
    batch_save_path = f'/content/drive/MyDrive/Colab Notebooks/embeddings_batch_{i+1}.npy'
    np.save(batch_save_path, embeddings)
    print(f"임베딩 데이터 {i+1}가 {batch_save_path}에 저장되었습니다.")
    
import numpy as np

# 저장된 임베딩 파일 경로
batch_file_paths = [
    '/content/drive/MyDrive/Colab Notebooks/embeddings_batch_1.npy',
    '/content/drive/MyDrive/Colab Notebooks/embeddings_batch_2.npy',
    '/content/drive/MyDrive/Colab Notebooks/embeddings_batch_3.npy'
]

# 각 파일을 로드하고 크기 확인
embeddings_list = []
for i, file_path in enumerate(batch_file_paths):
    embeddings = np.load(file_path)  # 파일 로드
    embeddings_list.append(embeddings)  # 리스트에 추가
    print(f"임베딩 데이터 {i+1} 크기: {embeddings.shape}")

# 로드된 임베딩 데이터 확인
# (필요 시) 임베딩 데이터의 일부 미리보기
for i, embeddings in enumerate(embeddings_list):
    print(f"임베딩 데이터 {i+1} 내용 미리보기:\n{embeddings[:5]}")  # 처음 5개 데이터 출력
    
# FAISS 인덱스 생성
dimension = embeddings_list[0].shape[1]

faiss_db = faiss.IndexFlatL2(dimension)

# 각 임베딩 데이터를 순차적으로 FAISS에 추가
for i, embeddings in enumerate(embeddings_list):
    faiss_db.add(embeddings.astype(np.float32))  # 임베딩 데이터를 float32 형식으로 변환하여 추가
    print(f"FAISS 인덱스에 {i+1}번째 데이터셋 추가 완료, 현재 저장된 벡터 개수: {faiss_db.ntotal}")
    
# FAISS 인덱스 파일 경로
faiss_index_path = '/content/drive/MyDrive/Colab Notebooks/faiss_index.index'

# FAISS 인덱스를 파일로 저장
faiss.write_index(faiss_db, faiss_index_path)
print(f"FAISS 인덱스가 {faiss_index_path}에 저장되었습니다.")

# FAISS 인덱스 파일 경로
faiss_index_path = '/content/drive/MyDrive/Colab Notebooks/faiss_index.index'

# FAISS 인덱스를 파일로 저장
faiss.write_index(faiss_db, faiss_index_path)
print(f"FAISS 인덱스가 {faiss_index_path}에 저장되었습니다.")

# FAISS 인덱스 파일 경로
faiss_index_path = '/content/drive/MyDrive/Colab Notebooks/faiss_index.index'

# FAISS 인덱스 로드
faiss_index = faiss.read_index(faiss_index_path)
print(f"FAISS 인덱스가 {faiss_index}에서 로드되었습니다.")

# 로드된 인덱스의 벡터 수 확인
print(f"저장된 벡터 개수: {faiss_index.ntotal}")

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
import faiss

# 임베딩 모델 로드 (예: 'jhgan/ko-sroberta-multitask')
model_embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')

# Google Generative AI API 설정
chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                                    temperature=0.2,  # 더 낮은 temperature로 설정해 할루시네이션 줄임
                                    top_p=0.85,        # top_p를 조정해 더 예측 가능한 답변 생성
                                    frequency_penalty=0.1  # 같은 단어의 반복을 줄이기 위해 패널티 추가
)

# 2. 멀티턴 대화를 위한 Memory 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 3. 멀티턴 프롬프트 템플릿 설정 (COT 방식 적용)
prompt_template = PromptTemplate(
    input_variables=["input_text", "search_results", "chat_history"],
    template="""
   ### 역할
    당신은 제주도 맛집과 관광지 추천 전문가입니다. 질문을 받을 때 논리적으로 생각한 후 단계별로 답변을 제공합니다. 복잡한 질문일수록 천천히 생각하고 적절한 데이터를 바탕으로 답변을 제공합니다.

    ### Chain of Thought 방식 적용:
    1. 사용자의 질문을 단계별로 분석합니다.
    2. 먼저 질문의 위치 정보를 파악합니다.
    3. 그 후에 사용자가 제공한 정보나 검색된 데이터를 바탕으로 관련성 있는 맛집과 관광지를 추천합니다.
    4. 단계를 나누어 정보를 체계적으로 제공합니다.

    ### 단계적 사고:
    1. 사용자 질문 분석
    2. 위치 정보 확인
    3. 관련 데이터 검색
    4. 추천 맛집 및 관광지 제공
    5. 추가 질문에 대한 친근한 대화 유지

    ### 지시사항
    당신은 사용자로부터 제주도의 맛집(식당, 카페 등)과 관광지를 추천하는 챗봇입니다.
    1. 사용자가 알고자 하는 동네(시군구)를 알려줄 때 까지 사용자에게 반문하세요. 이는 가장 중요합니다. 단, 위치를 두번 이상 반문하지 마세요. 만약 사용자가 위치를 모른다면 제일 평점이 좋은 3개의 식당+카페와 3개의 관광지를 안내해주세요.
    2. 친근하고 재미있으면서도 정겹게 안내하세요.
    3. source_id는 문서 번호입니다. 따라서 답변을 하는 경우 몇 번 문서를 인용했는지 답변 뒤에 언급하세요.
    4. 추천 할 때, 추천 이유와 소요되는 거리, 평점과 리뷰들도 보여줘. 만약 리뷰가 없는 곳이라면 ("작성된 리뷰가 없습니다.") 라고 해주세요.
    5. 4번의 지시사항과 함께 판매 메뉴 2개, 가격도 알려주세요.
    6. 만약 관광지와 식당이 구글검색에서 나오는 곳이면 지도(map)링크도 같이 첨부해줘. 지도 링크가 없는 곳은 지도 여부를 노출하지 말아주세요.
    7. 실제로 존재하는 식당과 관광지명을 추천해주어야 하며, %%흑돼지 맛집, 횟집 1 등 가게명이 명확하지 않은 답변은 하지 말아주세요.

    검색된 문서 내용:
    {search_results}

    대화 기록:
    {chat_history}

    사용자의 질문: {input_text}

    논리적인 사고 후 사용자에게 제공할 답변:
    """
)

# 4. 검색 및 응답 생성 함수
def search_faiss(query_embedding, k=5):
    """
    FAISS에서 유사한 벡터를 검색하여 원본 데이터 반환
    """
    # FAISS 인덱스에서 유사한 벡터 검색
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)

    # 검색된 인덱스를 바탕으로 원본 데이터 가져오기
    search_results = []
    total_length = 0  # 전체 길이 초기화

    for idx in indices[0]:
        found = False  # 찾은 데이터프레임 체크
        for df in dfs:
            if total_length + len(df) > idx:  # 현재 데이터프레임에서 유효한 인덱스인지 체크
                if idx - total_length >= 0 and idx - total_length < len(df):
                    search_results.append(df.iloc[idx - total_length])  # 인덱스 재조정
                found = True
                break
            total_length += len(df)  # 전체 길이에 데이터프레임 길이 추가
        if found:  # 이미 찾은 경우 더 이상 반복할 필요 없음
            continue

    return search_results




# 5. 대화형 응답 생성 함수 (COT 방식)
def generate_response(user_input):
    """
    사용자의 입력을 받아 FAISS 검색 후 응답 생성 (COT 적용)
    """
    # 사용자의 질문을 임베딩으로 변환
    query_embedding = model_embedding.encode([user_input])

    # FAISS 검색 수행
    search_results = search_faiss(query_embedding)

    # 검색된 결과를 텍스트 형식으로 변환
    search_results_str = "\n".join([result.to_string() for result in search_results])


    # PromptTemplate에 검색된 결과와 대화 기록 채우기
    filled_prompt = prompt_template.format(
        input_text=user_input,
        search_results=search_results_str,
        chat_history=memory.load_memory_variables({})["chat_history"]
    )

    # 1회 호출에서 5000 토큰 제한이므로 적절하게 텍스트를 나누어 처리
    response_parts = []
    while filled_prompt:
        # 최대 5000 토큰까지 잘라서 호출
        part = filled_prompt[:5000]
        filled_prompt = filled_prompt[5000:]

        # Google Generative AI API 호출 (대신 사용할 모델로 수정 가능)
        response = chat_model.invoke([{"role": "user", "content": part}])
        response_parts.append(response.content)

        # 호출 횟수 체크
        if len(response_parts) >= 3:
            break  # 최대 3회 호출 제한

    # 메모리에 대화 기록 저장
    for part in response_parts:
        memory.save_context({"input": user_input}, {"output": part})

    # 최종 응답 합치기
    return "\n".join(response_parts)

# 6. 챗봇 대화 루프
def chat():
    print("챗봇 대화를 시작합니다. 'exit'을 입력하면 종료됩니다.")
    while True:
        user_input = input("질문을 입력하세요: ")
        if user_input.lower() == "exit":
            break
        try:
            answer = generate_response(user_input)
            print("챗봇 응답:", answer)
        except Exception as e:
            print("오류 발생:", str(e))
            
# 챗봇 대화
chat()