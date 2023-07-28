# IMDB 영화 리뷰 데이터셋 로드
from tensorflow import keras
from keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)

# 정수형 데이터로 변환된 리뷰 단어 토큰
print(train_input.shape, train_target.shape)
print(len(train_input[0]))
print(train_input[0])
# 긍정/부정 0 또는 1 표시
print(train_target[:20])

# 검증셋 분리
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 각 리뷰 길이의 평균 및 중간값
import numpy as np

lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths))

# 각 리뷰 길이 별 출현 빈도
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel("length")
plt.ylabel("frequency")
plt.show()

# 리뷰의 대부분이 100자 이내이므로 100자로 리뷰들을 자르는 전처리 진행
from keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape)

# 전처리 진행 중 100자 보다 큰 경우 앞부분을 자르는 절차가 수행
print(train_seq[0])
print(train_input[0][-10:])

# 전처리 진행 중 100자 보다 작은 경우 앞부분에 0을 패딩하는 절차가 수행
print(train_seq[5])

# 검증셋 또한 전처리 진행
val_seq = pad_sequences(val_input, maxlen=100)

# 신경망 모델 생성
model = keras.Sequential()
# 순환 신경망 생성(활성화 함수는 tanh 적용)
# 입력셋의 정수값을 0000...1...00으로 나타나는 원-핫 인코딩으로 변환되어 (100, 500)크기의 데이터셋이 됨
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# 여기부터 로컬 GPU를 WSL에서 돌리는 것의 한계 발생
# 구글 코랩 사용 필요

# 입력값 원-핫 인코딩 적용
train_oh = keras.utils.to_categorical(train_seq)
val_oh = keras.utils.to_categorical(val_seq)

model.summary()

# 순환층 신경망 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_cb = keras.callbacks.ModelCheckpoint("best-simplernn-model.h5")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(
    train_oh,
    train_target,
    epochs=100,
    batch_size=64,
    validation_data=(val_oh, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# 훈련-검증 손실 그래프
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()

print(train_seq.nbytes, train_oh.nbytes)

# 단어를 임베딩으로 전환하는 방법
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation="sigmoid"))

model2.summary()

# 임베딩 모델 순환층 신경망 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])
checkpoint_cb = keras.callbacks.ModelCheckpoint("best-simplernn-model.h5")
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model2.fit(
    train_oh,
    train_target,
    epochs=100,
    batch_size=64,
    validation_data=(val_oh, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# 훈련-검증 손실 그래프
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["train", "val"])
plt.show()

print(train_seq.nbytes, train_oh.nbytes)
