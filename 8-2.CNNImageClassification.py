# 데이터셋 로드 및 훈련 준비
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (
    test_input,
    test_target,
) = keras.datasets.fashion_mnist.load_data()

# 3차원 => 4차원
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 합성곱 신경망 생성
model = keras.Sequential()
# 합성곱 층 1
model.add(
    keras.layers.Conv2D(
        32, kernel_size=3, activation="relu", padding="same", input_shape=(28, 28, 1)
    )
)
# 풀링 층 1
model.add(keras.layers.MaxPooling2D(2))
# 합성곱 층 2
model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"))
# 풀링 층 2
model.add(keras.layers.MaxPooling2D(2))
# 밀집 은닉 층 구성
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

# 신경망 구성 시각화
keras.utils.plot_model(model)
keras.utils.plot_model(model, show_shapes=True, to_file="cnn-architecture.png", dpi=300)

# 모델 컴파일 및 훈련
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
checkpoint_cb = keras.callbacks.ModelCheckpoint("best-cnn-model.h5")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(
    train_scaled,
    train_target,
    epochs=20,
    validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# 손실 그래프 출력
import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()

# 모델 평가
model.evaluate(val_scaled, val_target)

# 샘플 이미지 출력 및 예측과 대조
plt.imshow(val_scaled[0].reshape(28, 28), cmap="gray_r")
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)

# 해당 샘플의 클래스 막대그래프로 출력
plt.bar(range(1, 11), preds[0])
plt.xlabel("class")
plt.ylabel("prob.")
plt.show()

classes = ["티셔츠", "바지", "스웨터", "드레스", "코트", "샌달", "셔츠", "스니커즈", "가방", "앵클 부츠"]

import numpy as np

print(classes[np.argmax(preds)])

# 모든 테스트셋에 대해 일반화 검사
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

model.evaluate(test_scaled, test_target)
