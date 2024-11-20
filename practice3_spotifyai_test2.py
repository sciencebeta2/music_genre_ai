import pandas as pd # raw dataset
from surprise import SVD, accuracy # SVD model, 평가
from surprise import Reader, Dataset # SVD model의 dataset

rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/진짜최종.csv")
rating['user'].value_counts()
rating['genre'].value_counts()
tab = pd.crosstab(rating['user'], rating['genre'])

rating_g = rating.groupby(['user', 'genre'])
rating_g.sum()
tab = rating_g.sum().unstack() # 행렬구조로 변환

print(tab)

## test1에서 했던 "안본 영화들"을 행렬구조로 보여줌 -> 평점까지 추가로