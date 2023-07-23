import pandas as pd

# 와인 데이터셋 로드
wine = pd.read_csv("https://bit.ly/wine-date")
wine.head()
wine.info()
wine.describe()

# 와인 데이터셋에서 타겟셋 분리
data = wine[["alcohol", "sugar", "pH"]].to_numpy()
target = wine[["class"]].to_numpy()

# 훈련셋 / 테스트셋 분리
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)

print(train_input.shape, test_input.shape)

# 데이터 정규화
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀로 학습된 모델
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
print(lr.coef_, lr.intercept_)

# 결정 트리 방식으로 학습된 모델
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=["alcohol", "sugar", "pH"])
plt.show()

# 결정 트리의 깊이 3에서 가지치기한 결과
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=["alcohol", "sugar", "pH"])
plt.show()

# 결정 트리는 각 특성 별 정규화가 필요 없음
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=["alcohol", "sugar", "pH"])
plt.show()

# 2 1 3 순서로 특성 중요도를 파악
print(dt.feature_importances_)
