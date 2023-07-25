from tensorflow import keras

# 28*28 사이즈의 의류 이미지 60000개
(train_input, train_target), (
    test_input,
    test_target,
) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

# 교차 검증이 아닌 검증셋 분리
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

"""
    신경망 모델 생성 함수.
    parameter : a_layer
        별도 레이어를 받아 신경망에 추가
"""


def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation="relu"))

    if a_layer:
        model.add(a_layer)

    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


model = model_fn()
model.summary()
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

print(history.history.keys())

import matplotlib.pyplot as plt

# 신경망 모델 훈련 Epoch-Loss 지표 그래프 출력
plt.plot(history.history["loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 신경망 모델 훈련 Epoch-Accuracy 지표 그래프 출력
plt.plot(history.history["accuracy"])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

# Epoch 20회 설정 훈련 모델의 Epoch-Loss 지표
model = model_fn()
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)
plt.plot(history.history["loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 검증 손실 계산을 위한 검증셋 모델에 전달 & 지표 출력
model = model_fn()
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
history = model.fit(
    train_scaled,
    train_target,
    epochs=20,
    verbose=0,
    validation_data=(val_scaled, val_target),
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train, val"])
plt.show()

# Epoch 증가 시 Loss 증가율을 줄이기 위한 Optimizer 적용
model = model_fn()
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
history = model.fit(
    train_scaled,
    train_target,
    epochs=20,
    verbose=0,
    validation_data=(val_scaled, val_target),
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train, val"])
plt.show()

# 드롭아웃 적용
model = model_fn(keras.layers.Dropout(0.3))

model = model_fn()
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
history = model.fit(
    train_scaled,
    train_target,
    epochs=20,
    verbose=0,
    validation_data=(val_scaled, val_target),
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train, val"])
plt.show()

# Epoch 10으로 변경해 과대적합 방지
model = model_fn(keras.layers.Dropout(0.3))

model = model_fn()
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
history = model.fit(
    train_scaled,
    train_target,
    epochs=10,
    verbose=0,
    validation_data=(val_scaled, val_target),
)

# 모델 파일 저장
model.save_weights("model-weights.h5")
model.save("model-whole.h5")

# 모델 가중치 파일 로드
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights("model-weights.h5")

# 로드한 모델의 10개 샘플에 대한 정확도 체크
import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis=1)
print(np.mean(val_labels == val_target))

# 모델 전체 파일 로드
model = keras.models.load_model("model-whole.h5")
model.evaluate(val_scaled, val_target)

# 체크포인트 콜백 지정(최상의 검증 점수인 모델 저장)
model = model_fn(keras.layers.Dropout(0.3))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
checkpoint_cb = keras.callbacks.ModelCheckpoint("best-model.h5")
model.fit(
    train_scaled,
    train_target,
    epochs=10,
    verbose=0,
    validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb],
)

# 콜백으로 저장된 체크포인트 로드 & 평가
model = keras.models.load_model("best-model.h5")
model.evaluate(val_scaled, val_target)

# 체크포인트 콜백 + 조기 종료 콜백 지정
model = model_fn(keras.layers.Dropout(0.3))
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
checkpoint_cb = keras.callbacks.ModelCheckpoint("best-model.h5")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
model.fit(
    train_scaled,
    train_target,
    epochs=10,
    verbose=0,
    validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

print(early_stopping_cb)

# 최상의 손실일 경우에 중지한 모델 출력 및 평가
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train, val"])
plt.show()

model.evaluate(val_scaled, val_target)
