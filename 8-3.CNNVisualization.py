# 기존 최적 손실로 저장된 합성곱 신경망 모델 로드
from tensorflow import keras

model = keras.models.load_model("best-cnn-model.h5")

# 모델의 각 층과 첫번째 합성곱 층의 가중치
print(model.layers)
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)

# 첫 번째 합성곱 층의 가중치 넘파이로 변환
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())

# 각 가중치의 분포도 표시
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel("weight")
plt.ylabel("count")
plt.show()

# 첫 번째 합성곱 층의 커널 이미지 출력
fig, axs = plt.subplots(2, 16, figsize=(15, 2))

for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis("off")

plt.show()

# 학습되지 않은 빈 합성곱 모델 생성
no_train_model = keras.Sequential()
no_train_model.add(
    keras.layers.Conv2D(
        32, kernel_size=3, activation="relu", padding="same", input_shape=(28, 28, 1)
    )
)

# 해당 모델의 첫 번째 Conv층 가중치 저장
no_train_conv = no_train_model.layers[0]
print(no_train_conv.weights[0].shape)

# 해당 가중치의 평균 / 표준편차 분포
no_training_weights = no_train_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel("weight")
plt.ylabel("count")
plt.show()

# 빈 모델의 첫 번째 합성곱 층의 커널 이미지 출력
fig, axs = plt.subplots(2, 16, figsize=(15, 2))

for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis("off")
