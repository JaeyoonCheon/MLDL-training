from tensorflow import keras

# 28*28 사이즈의 의류 이미지 60000개
(train_input, train_target), (
    test_input,
    test_target,
) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)

# 교차 검증이 아닌 검증셋 분리
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 중간 100개 노드를 가진 시그모이드 은닉층
dense1 = keras.layers.Dense(100, activation="sigmoid", input_shape=(784,))
# 마지막 10개 노드를 가진 소프트맥스 출력층
dense2 = keras.layers.Dense(10, activation="softmax")

# 모델 생성 시 출력층은 항상 마지막에 위치
model = keras.Sequential([dense1, dense2])
model.summary()

# 모델 생성과 각각의 층을 한번에 만드는 방식
model = keras.Sequential(
    [
        keras.layers.Dense(
            100, activation="sigmoid", input_shape=(784,), name="hidden"
        ),
        keras.layers.Dense(10, activation="softmax", name="output"),
    ]
)
model.summary()

# add 메서드로 층을 추가하는 방식
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(keras.layers.Dense(10, activation="softmax", name="output"))
model.summary()

model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)

# ReLU 함수 은닉층 적용
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

# reshape를 적용하지 않은 훈련스케일셋 준비
(train_input, train_target), (
    test_input,
    test_target,
) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 앞서 ReLU 은닉층을 적용한 신경망 모델 피팅 & 평가
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)

# 모델 재생성
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.summary()

# adam Optimizer 적용
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)
