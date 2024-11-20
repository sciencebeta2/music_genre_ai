import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# CSV 데이터 불러오기
rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/진짜최종.csv")
rating['user'].value_counts()
rating['genre'].value_counts()
tab = pd.crosstab(rating['user'], rating['genre'])

print(tab)

# rating_g = rating.groupby(['user', 'genre'])
# rating_g.sum()
# tab = rating_g.sum().unstack() # 행렬구조로 변환
# # Reader 객체와 Dataset 객체로 변환
# reader = Reader(rating_scale=(1, 5))  # 평점 범위
# data = Dataset.load_from_df(df=rating, reader=reader)

# # 훈련셋과 테스트셋 분리
# trainset, testset = train_test_split(data, test_size=0.25)

# train = data.build_full_trainset() # 훈련셋
# test = train.build_testset() # 검정셋

# # 모델 설정
# model = SVD(n_factors=50, n_epochs=100, lr_all=0.009, reg_all=0.02, random_state=123)
# model.fit(trainset)  # 모델 훈련

# # 예시로 추천할 항목들 설정
# user_id = 'B'  # 추천대상자
# item_ids = ['ballard', 'hiphop', 'k-pop']  # 추천할 항목들
# actual_rating = 0  # 실제 평점은 알지 못하므로 0으로 설정

# # 추천 예측
# for item_id in item_ids:
#     prediction = model.predict(user_id, item_id, actual_rating)
#     print(f"user: {user_id}    item: {item_id}    r_ui = {prediction.r_ui}   est = {prediction.est}")
