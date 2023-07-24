import numpy as np
import matplotlib.pyplot as plt

# 과일 사진 데이터 로드
fruits = np.load("fruits_300.npy")
fruits_2d = fruits.reshape(-1, 100 * 100)

# 클러스터링 된 군집 이미지 레이블 별 표시
import matplotlib.pyplot as plt


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * ratio, rows * ratio), squeeze=False
    )

    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap="gray_r")
            axs[i, j].axis("off")

    plt.show()


# 주성분 분석
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape)

# 50개의 주성분 벡터를 분산이 큰 순서대로 나열
draw_fruits(pca.components_.reshape(-1, 100, 100))

print(fruits_2d.shape)

# 10000개(100*100 픽셀)의 특성을 가진 배열에서 50개의 주성분으로 축소
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# 주성분 50개로부터 다시 역함수를 걸어 10000개의 특성을 복원
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)

for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start : start + 100])
    print("\n")

# 주성분이 가진 원본 데이터의 분산 값을 '설명된 분산'으로 표시
print(np.sum(pca.explained_variance_ratio_))
plt.plot(pca.explained_variance_ratio_)

# 로지스틱 회귀 모델로 원 데이터셋과 차원 축소된 데이터셋을 비교
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

target = np.array([0] * 100 + [1] * 100 + [2] * 100)

from sklearn.model_selection import cross_validate

# 원본 10000개 교차 검증
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores["test_score"]))
print(np.mean(scores["fit_time"]))

# 50개 주성분 교차 검증
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores["test_score"]))
print(np.mean(scores["fit_time"]))

# 분산 50%에 달하는 주성분 선택
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

print(pca.n_components_)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# 분산 50%인 2개 주성분 교차 검증
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores["test_score"]))
print(np.mean(scores["fit_time"]))

# K-Means 알고리즘으로 클러스터 탐색
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

# 클러스터링 된 레이블로 과일 이미지 출력
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

# 앞서 선택한 2개 특성을 바탕으로 클러스터 산점도 표시
for label in range(3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])

plt.legend(["apple", "banana", "pineapple"])
plt.show()
