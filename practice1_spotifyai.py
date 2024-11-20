# # # import pandas as pd
# # # from pandas import Series

# # # price = pd.Series([4000, 3000, 3500, 2000])
# # # person = Series({'name' : '홍길동', 'age' : 35, 'addr' : '서울시'})
# # # print(person)

# # # # dtype 이거 어케해야됨? https://codesample-factory.tistory.com/1047

# # #### 2024-11-06 ####
# # #실습 전 준비사항 - 아나콘다 프롬프트에서 'Surprise' 패키지 설치하기 ( "pip install scikit-surprise" 입력)

# # import pandas as pd # raw dataset
# # from surprise import SVD, accuracy # SVD model, 평가
# # from surprise import Reader, Dataset # SVD model의 dataset
# # from surprise import Dataset, Reader, accuracy, SVD
# # from surprise.model_selection import train_test_split

# # rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/movie_rating.csv")
# # rating.head()   #   critic(user)   title(item)   rating

# # rating['critic'].value_counts()
# # rating['title'].value_counts()
# # tab = pd.crosstab(rating['critic'], rating['title'])

# # reader = Reader(rating_scale= (1, 5)) # 평점 범위
# # data = Dataset.load_from_df(df=rating, reader=reader)

# # train = data.build_full_trainset() # 훈련셋
# # test = train.build_testset() # 검정셋

# # help(SVD)

# # model = SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, random_state=None, verbose=False)
# # model.fit(train) # model 생성

# # user_id = 'Toby' # 추천대상자
# # item_ids = ['The Night', 'Just My', 'Lady'] # 추천 대상 영화
# # actual_rating = 0 # 실제 평점


# # for item_id in item_ids :
# #     print(model.predict(user_id, item_id, actual_rating))

# # # UnicodeEscape Error 복구사이트 : https://paperlover.tistory.com/22

# #### 2024-11-12 ####

# import pandas as pd # raw dataset
# from surprise import SVD, accuracy # SVD model, 평가
# from surprise import Reader, Dataset # SVD model의 dataset

# import pandas as pd
# from surprise import SVD, Dataset, Reader
# from surprise.model_selection import train_test_split
# from surprise import accuracy

# rating = pd.read_csv("C:/Users/scien/OneDrive/바탕 화면/movie_rating.csv")


# # # 예시 데이터 (userID, itemID, rating)
# # data = [
# #     (1, 'item1', 5),
# #     (1, 'item2', 3),
# #     (2, 'item1', 4),
# #     (2, 'item2', 2),
# #     (3, 'item1', 3),
# #     (3, 'item3', 4),
# # ]

# # 데이터가 리스트 형식이므로, 'surprise'에서 사용할 수 있도록 변환합니다.
# reader = Reader(rating_scale=(1, 5))  # 평점이 1에서 5까지의 범위라고 가정
# dataset = Dataset.load_from_df(pd.DataFrame(data, columns=['userID', 'itemID', 'rating']), reader)
# # 리스트 형태의 데이터를 사용할 수 있는 형식으로 변경

# trainset, testset = train_test_split(dataset, test_size=0.2)
# # 추출한 데이터셋을 훈련 세트와 테스트 세트로 분리하고, 2 : 8의 비율로 테스트세트:훈련세트로 나눈 후 testset, trainset이라는 데이터값을 반환한다.


# # SVD 모델 생성
# svd = SVD()

# # 모델 훈련
# svd.fit(trainset)

# # 테스트 데이터로 예측
# predictions = svd.test(testset)

# # 예측 정확도 평가 (RMSE)
# rmse = accuracy.rmse(predictions)
# print(f"RMSE: {rmse}")

# # 특정 사용자에 대해 예측을 수행 (예: userID 1, itemID 'item3'에 대해 예측)
# prediction = svd.predict(1, 'item3')
# print(f"Prediction for user 1 on item 'item3': {prediction}")

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

reader = Reader(rating_scale= (1, 5)) # 평점 범위
data = Dataset.load_from_df(df=rating, reader=reader)

train = data.build_full_trainset() # 훈련셋
test = train.build_testset() # 검정셋

# help(SVD) #이거 때매 SVD 설명이 떴다;;;

model = SVD(n_factors=50, n_epochs=100, lr_all=0.009, reg_all=0.82, random_state=123)
model.fit(train) # model 생성

user_id = 'A' # 추천대상자
item_ids = ['ballard', 'j-pop', 'k-pop'] # 추천 대상 영화
actual_rating = 0 # 실제 평점

 
for item_id in item_ids:
    prediction = model.predict(user_id, item_id, actual_rating)
    print(f"user: {user_id}    item: {item_id}    r_ui = {prediction.r_ui}   est = {prediction.est}")

# 여기서 결괏값 출력하기 전에 나오는 
# # class SVD(surprise.prediction_algorithms.algo_base.AlgoBase)
#  |  SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, random_state=None, verbose=False)
#  |
#  |  The famous *SVD* algorithm, as popularized by `Simon Funk
#  |  <https://sifter.org/~simon/journal/20061211.html>`_ during the Netflix
#  |  Prize. When baselines are not used, this is equivalent to Probabilistic
#  |  Matrix Factorization :cite:`salakhutdinov2008a` (see :ref:`note
#  |  <unbiased_note>` below).
#  |
#  |  The prediction :math:`\hat{r}_{ui}` is set as:
#  |
#  |  .. math::
#  |      \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u
#  |
#  |  If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
#  |  :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
#  |  with :math:`b_i` and :math:`q_i`.
#  |
#  |  For details, see equation (5) from :cite:`Koren:2009`. See also
#  |  :cite:`Ricci:2010`, section 5.3.1.
#  |
#  |  To estimate all the unknown, we minimize the following regularized squared
#  |  error:
#  |
#  |  .. math::
#  |      \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \right)^2 +
#  |      \lambda\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\right)
#  |
#  |
#  |  The minimization is performed by a very straightforward stochastic gradient
#  |  descent:
#  |      b_i &\leftarrow b_i &+ \gamma (e_{ui} - \lambda b_i)\\
#  |      p_u &\leftarrow p_u &+ \gamma (e_{ui} \cdot q_i - \lambda p_u)\\
#  |      q_i &\leftarrow q_i &+ \gamma (e_{ui} \cdot p_u - \lambda q_i)
#  |
#  |  where :math:`e_{ui} = r_{ui} - \hat{r}_{ui}`. These steps are performed
#  |  over all the ratings of the trainset and repeated ``n_epochs`` times.
#  |  Baselines are initialized to ``0``. User and item factors are randomly
#  |  initialized according to a normal distribution, which can be tuned using
#  |  the ``init_mean`` and ``init_std_dev`` parameters.
#  |
#  |  You also have control over the learning rate :math:`\gamma` and the
#  |  regularization term :math:`\lambda`. Both can be different for each
#  |  kind of parameter (see below). By default, learning rates are set to
#  |  ``0.005`` and regularization terms are set to ``0.02``.
#  |
#  |  .. _unbiased_note:
#  |
#  |  .. note::
#  |      You can choose to use an unbiased version of this algorithm, simply
#  |      predicting:
#  |
#  |      .. math::
#  |          \hat{r}_{ui} = q_i^Tp_u
#  |
#  |      This is equivalent to Probabilistic Matrix Factorization
#  |      (:cite:`salakhutdinov2008a`, section 2) and can be achieved by setting
#  |      the ``biased`` parameter to ``False``.
#  |
#  |
#  |  Args:
#  |      n_factors: The number of factors. Default is ``100``.
#  |      n_epochs: The number of iteration of the SGD procedure. Default is
#  |          ``20``.
#  |      biased(bool): Whether to use baselines (or biases). See :ref:`note
#  |      init_std_dev: The standard deviation of the normal distribution for
#  |          factor vectors initialization. Default is ``0.1``.
#  |      lr_all: The learning rate for all parameters. Default is ``0.005``.
#  |      reg_all: The regularization term for all parameters. Default is
#  |          ``0.02``.
#  |      lr_bu: The learning rate for :math:`b_u`. Takes precedence over
#  |          ``lr_all`` if set. Default is ``None``.
#  |      lr_bi: The learning rate for :math:`b_i`. Takes precedence over
#  |          ``lr_all`` if set. Default is ``None``.
#  |      lr_pu: The learning rate for :math:`p_u`. Takes precedence over
#  |          ``lr_all`` if set. Default is ``None``.
#  |      lr_qi: The learning rate for :math:`q_i`. Takes precedence over
#  |          ``lr_all`` if set. Default is ``None``.
#  |      reg_bu: The regularization term for :math:`b_u`. Takes precedence
#  |          over ``reg_all`` if set. Default is ``None``.
#  |      reg_bi: The regularization term for :math:`b_i`. Takes precedence
#  |          over ``reg_all`` if set. Default is ``None``.
#  |      reg_pu: The regularization term for :math:`p_u`. Takes precedence
#  |          over ``reg_all`` if set. Default is ``None``.
#  |      reg_qi: The regularization term for :math:`q_i`. Takes precedence
#  |          over ``reg_all`` if set. Default is ``None``.
#  |      random_state(int, RandomState instance from numpy, or ``None``):
#  |          Determines the RNG that will be used for initialization. If
#  |          int, ``random_state`` will be used as a seed for a new RNG. This is
#  |          useful to get the same initialization over multiple calls to
#  |          ``fit()``.  If RandomState instance, this same instance is used as
#  |          RNG. If ``None``, the current RNG from numpy is used.  Default is
#  |          ``None``.
#  |      verbose: If ``True``, prints the current epoch. Default is ``False``.
#  |
#  |  Attributes:
#  |      pu(numpy array of size (n_users, n_factors)): The user factors (only
#  |          exists if ``fit()`` has been called)
#  |      qi(numpy array of size (n_items, n_factors)): The item factors (only
#  |      bi(numpy array of size (n_items)): The item biases (only
#  |          exists if ``fit()`` has been called)
#  |
#  |  Method resolution order:
#  |      SVD
#  |      surprise.prediction_algorithms.algo_base.AlgoBase
#  |      builtins.object
#  |
#  |  Methods defined here:
#  |
#  |  __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=0.1, lr_all=0.005, reg_all=0.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, random_state=None, verbose=False)
#  |
#  |  estimate(self, u, i)
#  |
#  |  fit(self, trainset)
#  |
#  |  sgd(self, trainset)
#  |
#  |  ----------------------------------------------------------------------
#  |  Methods inherited from surprise.prediction_algorithms.algo_base.AlgoBase:
#  |
#  |  compute_baselines(self)
#  |      Compute users and items baselines.
#  |
#  |      The way baselines are computed depends on the ``bsl_options`` parameter
#  |      passed at the creation of the algorithm (see
#  |      :ref:`baseline_estimates_configuration`).
#  |
#  |      This method is only relevant for algorithms using :func:`Pearson
#  |      baseline similarity<surprise.similarities.pearson_baseline>` or the
#  |      :class:`BaselineOnly
#  |      <surprise.prediction_algorithms.baseline_only.BaselineOnly>` algorithm.
#  |
#  |  compute_similarities(self)
#  |      Build the similarity matrix.
#  |
#  |      The way the similarity matrix is computed depends on the
#  |      ``sim_options`` parameter passed at the creation of the algorithm (see
#  |      :ref:`similarity_measures_configuration`).
#  |
#  |      This method is only relevant for algorithms using a similarity measure,
#  |      such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.
#  |
#  |      Returns:
#  |          The similarity matrix.
#  |
#  |  default_prediction(self)
#  |      Used when the ``PredictionImpossible`` exception is raised during a
#  |      call to :meth:`predict()
#  |      <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
#  |      default, return the global mean of all ratings (can be overridden in
#  |      child classes).
#  |
#  |      Returns:
#  |          (float): The mean of all ratings in the trainset.
#  |
#  |  get_neighbors(self, iid, k)
#  |      Return the ``k`` nearest neighbors of ``iid``, which is the inner id
#  |      of a user or an item, depending on the ``user_based`` field of
#  |      ``sim_options`` (see :ref:`similarity_measures_configuration`).
#  |
#  |      As the similarities are computed on the basis of a similarity measure,
#  |      this method is only relevant for algorithms using a similarity measure,
#  |      such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.
#  |
#  |      For a usage example, see the :ref:`FAQ <get_k_nearest_neighbors>`.
#  |
#  |
#  |          k(int): The number of neighbors to retrieve.
#  |
#  |      Returns:
#  |          The list of the ``k`` (inner) ids of the closest users (or items)
#  |          to ``iid``.
#  |
#  |  predict(self, uid, iid, r_ui=None, clip=True, verbose=False)
#  |      Compute the rating prediction for given user and item.
#  |
#  |      The ``predict`` method converts raw ids to inner ids and then calls the
#  |      ``estimate`` method which is defined in every derived class. If the
#  |      prediction is impossible (e.g. because the user and/or the item is
#  |      unknown), the prediction is set according to
#  |      :meth:`default_prediction()
#  |      <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.
#  |
#  |      Args:
#  |          uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
#  |          iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
#  |          r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
#  |              ``None``.
#  |          clip(bool): Whether to clip the estimation into the rating scale.
#  |              For example, if :math:`\hat{r}_{ui}` is :math:`5.5` while the
#  |              rating scale is :math:`[1, 5]`, then :math:`\hat{r}_{ui}` is
#  |              set to :math:`5`. Same goes if :math:`\hat{r}_{ui} < 1`.
#  |              Default is ``True``.
#  |          verbose(bool): Whether to print details of the prediction.  Default
#  |              is False.
#  |
#  |      Returns:
#  |          A :obj:`Prediction            <surprise.prediction_algorithms.predictions.Prediction>` object
#  |          containing:
#  |
#  |          - The estimated rating (:math:`\hat{r}_{ui}`).
#  |          - Some additional details about the prediction that might be useful
#  |            for later analysis.
#  |
#  |  test(self, testset, verbose=False)
#  |      Test the algorithm on given testset, i.e. estimate all the ratings
#  |      in the given testset.
#  |
#  |      Args:
#  |          testset: A test set, as returned by a :ref:`cross-validation
#  |              itertor<use_cross_validation_iterators>` or by the
#  |              :meth:`build_testset() <surprise.Trainset.build_testset>`
#  |              method.
#  |          verbose(bool): Whether to print details for each predictions.
#  |              Default is False.
#  |
#  |      Returns:
#  |          A list of :class:`Prediction            <surprise.prediction_algorithms.predictions.Prediction>` objects
#  |          that contains all the estimated ratings.
#  |
#  |  ----------------------------------------------------------------------
#  |  Data descriptors inherited from surprise.prediction_algorithms.algo_base.AlgoBase:
#  |
#  |  __dict__
#  |      dictionary for instance variables
#  |
#  |  __weakref__
#  |      list of weak references to the object
# 이거는 계속 누르면 됨 다 보고