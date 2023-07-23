import pandas as pd

# 물고기 데이터 로드(종, 무게, 길이, 대각 길이, 높이, 너비)
fish = pd.read_csv("https://bit.ly/fish_csv")
fish.head()

print(pd.unique(fish["Species"]))

# 종을 제외한 나머지 특성만 분리
fish_input = fish[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()

print(fish_input[:5])

# 종명이 곧 타겟값이므로 분리
fish_target = fish["Species"].to_numpy()

# 훈련셋 + 테스트셋으로 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

# 정규화
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# K-최근접 이웃 분류법으로 분류하는 경우
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))

import numpy as np

# 예측값이 임의 클래스일 확률을 계산
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# 4번째 이웃의 거리 및 인덱스
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

# 도미 / 빙어의 이진 분류인 경우
# boolean indexing을 사용해 도미 & 빙어인 경우만 분리
bream_smelt_indexes = (train_target == "Bream") | (train_target == "Smelt")
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 선형 회귀로 모델 훈련 및 평가
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5]))

print(lr.classes_)
# 회귀 선형 방정식의 계수와 절편(각 특성 별 가중치)
print(lr.coef_, lr.intercept_)

# 회귀 선형 방정식(가중치 * 특성1 + 가중치 * 특성2 + ...)의 결과인 z값을 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit

# 시그모이드 함수로 값을 0~1 사이의 값으로 변환
print(expit(decisions))

# 로지스틱 회귀로 다중 분류(모든 각 특성 적용)
# L2 규제 C=20
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

print(lr.classes_)
print(lr.coef_.shape, lr.intercept_.shape)

# 5개 샘플에 대한 클래스 7개 각각의 z값
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

from scipy.special import softmax

# 각 값들의 합이 1이 되는 소프트맥스 함수를 통해 0~1 사이의 값으로 변환
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
