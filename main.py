import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset

rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/진짜최종.csv")
rating['user'].value_counts()
rating['genre'].value_counts()
tab = pd.crosstab(rating['user'], rating['genre'])

rating_g = rating.groupby(['user', 'genre'])
rating_g.sum()
tab = rating_g.sum().unstack()

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df=rating, reader=reader)

trainset, testset = train_test_split(data, test_size=0.25)

train = data.build_full_trainset()
test = train.build_testset()

model = SVD(n_factors=50, n_epochs=100, lr_all=0.009, reg_all=0.02, random_state=123)
model.fit(trainset)

user_id = 'C'
item_ids = ['ballard', 'hiphop', 'k-pop']
actual_rating = 0

for item_id in item_ids:
    prediction = model.predict(user_id, item_id, actual_rating)
    print(f"user: {user_id}    item: {item_id}    r_ui = {prediction.r_ui}   est = {prediction.est}")
