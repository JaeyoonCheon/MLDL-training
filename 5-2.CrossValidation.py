import pandas as pd

# 와인 데이터셋 로드
wine = pd.read_csv("https://bit.ly/wine-date")

# 와인 데이터셋에서 타겟셋 분리
data = wine[["alcohol", "sugar", "pH"]].to_numpy()
target = wine[["class"]].to_numpy()

# 훈련셋 / 테스트셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# 훈련 셋을 다시 훈련셋과 검증셋으로 분리
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)
print(sub_input.shape, val_input.shape)

# 결정 트리로 학습한 모델의 검증 결과
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

# 전체 데이터셋을 n개로 쪼개 각각을 검증셋으로 설정한 결과들을 평균하는 교차 검증
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
# n번째를 검증셋으로 사용해 학습된 결과 점수
print(scores)
# 모든 결과 점수를 평균낸 검증 점수
import numpy as np

print(np.mean(scores["test_score"]))

# 훈련셋을 셔플하기 위한 분할기
from sklearn.model_selection import StratifiedKFold

# 이전 코드와 동일한 경우
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores["test_score"]))

# 10-Fold 교차 검증
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores["test_score"]))

# 하이퍼 파라미터 탐색을 위한 그리드 서치
from sklearn.model_selection import GridSearchCV

# min_impurity_decrease라는 하이퍼 파라미터에 대해 5개의 값을 5-Fold 검증 그리드 서치
params = {"min_impurity_decrease": [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)

best_index = np.argmax(gs.cv_results_["mean_test_score"])
print(gs.cv_results_["params"][best_index])

# 하이퍼 파라미터 추가
params = {
    "min_impurity_decrease": np.arange(0.0001, 0.001, 0.0001),
    "max_depth": range(5, 20, 1),
    "min_samples_split": range(2, 100, 10),
}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

print(gs.best_params_)
print(gs.cv_results_["mean_test_score"])

# 랜덤 서치
from scipy.stats import uniform, randint

params = {
    "min_impurity_decrease": uniform(0.0001, 0.001),
    "max_depth": randint(20, 50),
    "min_samples_split": randint(2, 25),
    "min_samples_leaf": randint(1, 25),
}

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    n_iter=100,
    n_jobs=-1,
    random_state=42,
)
gs.fit(train_input, train_target)

print(gs.best_params_)
print(np.max(gs.cv_results_["mean_test_score"]))
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
