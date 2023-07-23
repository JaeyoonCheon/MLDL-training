import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# RandomForest 분류
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(
    rf, train_input, train_target, return_train_score=True, n_jobs=-1
)
print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

rf.fit(train_input, train_target)
print(rf.feature_importances_)

# 모델 평가
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)

# 엑스트라 트리
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(
    et, train_input, train_target, return_train_score=True, n_jobs=-1
)
print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

et.fit(train_input, train_target)
print(et.feature_importances_)

# 그레디언트 부스팅
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(
    gb, train_input, train_target, return_train_score=True, n_jobs=-1
)
print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

# 학습률과 트리 개수를 증가시킨 결과
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(
    gb, train_input, train_target, return_train_score=True, n_jobs=-1
)
print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

gb.fit(train_input, train_target)
print(gb.feature_importances_)

# 히스토그램 기반 그레디언트 부스팅
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(
    hgb, train_input, train_target, return_train_score=True, n_jobs=-1
)
print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))

hgb.fit(train_input, train_target)
print(hgb.feature_importances_)
print(hgb.score(test_input, test_target))
