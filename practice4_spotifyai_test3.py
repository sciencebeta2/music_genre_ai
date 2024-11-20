import pandas as pd # raw dataset
from surprise import SVD, accuracy # SVD model, 평가
from surprise import Reader, Dataset # SVD model의 dataset


rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/music_sample.csv")
rating['user'].value_counts()
rating['genre'].value_counts()
tab = pd.crosstab(rating['user'], rating['genre'])

rating_g = rating.groupby(['user', 'genre'])
rating_g.sum()
tab = rating_g.sum().unstack() # 행렬구조로 변환

reader = Reader(rating_scale= (1, 5)) # 평점 범위
data = Dataset.load_from_df(df=rating, reader=reader)

train = data.build_full_trainset() # 훈련셋
test = train.build_testset() # 검정셋

help(SVD)

model = SVD(n_factors=100, n_epochs=20, random_state=123)
model.fit(train) # model 생성

user_id = 'A' # 추천대상자
item_ids = ['ballard', 'j-pop', 'k-pop'] # 추천 대상 영화
actual_rating = 0 # 실제 평점

 

for item_id in item_ids :
    print(model.predict(user_id, item_id, actual_rating))