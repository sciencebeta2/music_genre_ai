import pandas as pd # raw dataset
from surprise import SVD, accuracy # SVD model, 평가
from surprise import Reader, Dataset # SVD model의 dataset


rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/진짜최종.csv")
rating['user'].value_counts()
rating['genre'].value_counts()
tab = pd.crosstab(rating['user'], rating['genre'])

print(tab)

## 이건 영화를 본 횟수를 출력한다.