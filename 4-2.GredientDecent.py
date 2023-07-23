import pandas as pd

# 물고기 데이터 로드(종, 무게, 길이, 대각 길이, 높이, 너비)
fish = pd.read_csv("https://bit.ly/fish_csv")
fish.head()

# 종을 제외한 나머지 특성만 분리
fish_input = fish[["Weight", "Length", "Diagonal", "Height", "Width"]].to_numpy()

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

# 확률적 경사하강법 분류
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 1 Epoch 추가 실행
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# 300번의 Epoch 시행 시 훈련셋과 테스트셋의 정확도 비교
import numpy as np

sc = SGDClassifier(loss="log_loss", random_state=42)

train_score = []
test_score = []

classes = np.unique(train_target)

for _ in range(300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.show()

# 100번의 Epoch가 최적의 반복 횟수로 간주되므로, 100번으로 Epoch를 지정하여 fitting
sc = SGDClassifier(loss="log_loss", max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# Hinge 손실함수(서포트 벡터 머신)를 사용한 확률 경사하강법
sc = SGDClassifier(loss="hinge", max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))
