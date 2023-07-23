import numpy as np
import matplotlib.pyplot as plt

# 과일 사진 데이터 로드
fruits = np.load("fruits_300.npy")
fruits_2d = fruits.reshape(-1, 100 * 100)

# K-평균 알고리즘
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

print(km.labels_)
print(np.unique(km.labels_, return_counts=True))

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


draw_fruits(fruits[km.labels_ == 0])
draw_fruits(fruits[km.labels_ == 1])
draw_fruits(fruits[km.labels_ == 2])

# 각 클러스터링된 레이블 별 클러스터 중심 이미지
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# 개별 데이터의 클러스터 중심까지의 거리
print(km.transform(fruits_2d[100:101]))
print(km.predict(fruits_2d[100:101]))

draw_fruits(fruits[100:101])
print(km.n_iter_)
