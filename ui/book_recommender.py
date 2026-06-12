"""
키워드 기반 검색결과 재배열 및 유사 도서 추천 모듈 (TF-IDF only)

사용 모델:
- models/tfidf_total.pkl          : 학습된 TfidfVectorizer
- models/Tfidf_book_total.mtx     : 전체 도서(4938권)의 TF-IDF 행렬 (csv row 순서와 동일)

기능:
1. rank_by_keyword(keyword, indices)
   - 입력 키워드를 TF-IDF 벡터로 변환
   - 주어진 도서 인덱스(검색 결과) 내에서 키워드와의 코사인 유사도로 재정렬

2. recommend_similar_books(book_index, top_n)
   - 특정 책(csv row index)의 TF-IDF 벡터와 다른 책들의 코사인 유사도를 계산
   - 유사도가 높은 top_n개의 책 인덱스를 반환
"""

import pickle
import numpy as np
from scipy.io import mmread
from sklearn.metrics.pairwise import cosine_similarity

TFIDF_VECTORIZER_PATH = "../models/tfidf_total.pkl"
TFIDF_MATRIX_PATH = "../models/Tfidf_book_total.mtx"


class BookRecommender:
    def __init__(self):
        # TF-IDF 벡터라이저
        with open(TFIDF_VECTORIZER_PATH, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        # 전체 도서 TF-IDF 행렬 (csv row 순서와 1:1 대응)
        self.tfidf_matrix = mmread(TFIDF_MATRIX_PATH).tocsr()

    def rank_by_keyword(self, keyword: str, indices: list) -> list:
        """
        검색 결과(indices: csv row index 리스트)를 키워드와의
        TF-IDF 코사인 유사도 기준 내림차순으로 재정렬합니다.

        Args:
            keyword: 사용자가 입력한 키워드
            indices: 현재 검색 결과의 csv row index 리스트 (df.index 기준, 0-base)

        Returns:
            list: 유사도 순으로 재정렬된 csv row index 리스트
        """
        if not keyword.strip() or not indices:
            return indices

        query_vec = self.tfidf_vectorizer.transform([keyword])

        subset_matrix = self.tfidf_matrix[indices]
        sims = cosine_similarity(query_vec, subset_matrix).flatten()

        order = np.argsort(sims)[::-1]
        ranked_indices = [indices[i] for i in order]

        return ranked_indices

    def recommend_similar_books(self, book_index: int, top_n: int = 5, exclude_self: bool = True) -> list:
        """
        특정 책(csv row index)과 TF-IDF 코사인 유사도가 높은 책들의 인덱스를 반환합니다.

        Args:
            book_index: 기준 책의 csv row index (0-base)
            top_n: 추천할 책 개수
            exclude_self: 자기 자신을 결과에서 제외할지 여부

        Returns:
            list: 유사도가 높은 책들의 csv row index 리스트 (길이 top_n)
        """
        target_vec = self.tfidf_matrix[book_index]
        sims = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        order = np.argsort(sims)[::-1]

        if exclude_self:
            order = order[order != book_index]

        return order[:top_n].tolist()


if __name__ == "__main__":
    recommender = BookRecommender()

    # 사용 예시 1: 키워드로 재정렬
    sample_indices = list(range(0, 20))  # 검색 결과라고 가정
    ranked = recommender.rank_by_keyword("추리", sample_indices)
    print("키워드 '추리' 기준 재정렬:", ranked[:5])

    # 사용 예시 2: 유사 도서 추천
    similar = recommender.recommend_similar_books(book_index=0, top_n=5)
    print("0번 책(이방인)과 유사한 책 인덱스:", similar)