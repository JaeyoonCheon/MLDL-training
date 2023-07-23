import numpy as np
import matplotlib.pyplot as plt

# 과일 사진 데이터 로드
fruits = np.load("fruits_300.npy")

print(fruits.shape)
print(fruits[0, 0, :])

# 0 ~ 255 Grayscale 색상의 100*100 크기의 배열 사진 300개
plt.imshow(fruits[0], cmap="gray_r")
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap="gray_r")
axs[1].imshow(fruits[200], cmap="gray_r")
plt.show()

# 2차원 배열 형태의 사진 데이터를 1차원 배열 형태로 전환
apple = fruits[:100].reshape(-1, 100 * 100)
pineapple = fruits[100:200].reshape(-1, 100 * 100)
banana = fruits[200:300].reshape(-1, 100 * 100)

print(apple.shape)
print(apple.mean(axis=1))

# 데이터 분류 별 픽셀 색상 데이터 평균 히스토그램
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(["apple", "pineapple", "banana"])
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

# 각 분류 별 100장의 픽셀 각각의 평균값을 가지는 이미지
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap="gray_r")
axs[1].imshow(pineapple_mean, cmap="gray_r")
axs[2].imshow(banana_mean, cmap="gray_r")
plt.show()

# 각 분류 별 클래스의 평균값과 가까운 값을 가지는 사진 선택 (사과의 경우)
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))

for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap="gray_r")
        axs[i, j].axis("off")

plt.show()
