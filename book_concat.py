import pandas as pd

# 파일별 장르 정보
file_infos = [
    {
        'path': './data/yes24_steadyseller_SF.csv',
        'genre': 'SF'
    },
    {
        'path': './data/yes24_steadyseller_공포_스릴러.csv',
        'genre': '공포, 스릴러'
    },
    {
        'path': './data/yes24_steadyseller_추리_미스터리.csv',
        'genre': '추리, 미스터리'
    },
    {
        'path': './data/yes24_steadyseller_로맨스.csv',
        'genre': '로맨스'
    },
    {
        'path': './data/yes24_steadyseller_역사.csv',
        'genre': '역사'
    },
    {
        'path': './data/yes24_steadyseller_판타지.csv',
        'genre': '판타지'
    }
]

# 최종 데이터에서 남길 컬럼
use_columns = [
    'ISBN',
    '상품명',
    '저자',
    '출판사',
    '설명'
]

df_list = []

for info in file_infos:
    # CSV 읽기
    df = pd.read_csv(
        info['path'],
        encoding='utf-8-sig',
        dtype={'ISBN': str}
    )

    # 필요한 컬럼만 남기기
    df = df[use_columns]

    # 장르 컬럼 추가
    df['장르'] = info['genre']

    # 리스트에 추가
    df_list.append(df)

# 여러 장르 CSV를 하나로 합치기
df_books = pd.concat(df_list, ignore_index=True)

# 결측치 처리
df_books = df_books.fillna('')

# 같은 책이 여러 장르에 있을 경우 장르를 합치기
df_books = df_books.groupby(
    ['ISBN', '상품명', '저자', '출판사', '설명'],
    as_index=False
)['장르'].agg(lambda x: ', '.join(sorted(set(x))))

# 추천 시스템용 통합 텍스트 컬럼 생성
#df_books['contents'] = (
#    df_books['상품명'] + ' ' +
#    df_books['저자'] + ' ' +
#    df_books['출판사'] + ' ' +
#    df_books['설명'] + ' ' +
#    df_books['장르']
#)

# 저장
df_books.to_csv(
    './data/yes24_steadyseller_장르통합.csv',
    index=False,
    encoding='utf-8-sig'
)

print(df_books.info())
print(df_books.head())