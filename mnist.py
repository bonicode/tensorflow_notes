import tensorflow as tf
from tensorflow.keras import Sequential, layers
import tensorflow_datasets as tfds

# this is used if there is train test split.
(train_ds, test_ds), info_ds = tfds.load(
    "mnist",
    as_supervised=True,
    shuffle_files=True,
    split=["train", "test"],
    with_info=True,
)

# This is to show some image sample
# fig = tfds.show_examples(train_ds, ds_info=info_ds, rows=4, cols=4)


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCHSIZE = 64

train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(info_ds.splits["train"].num_examples)
train_ds = train_ds.batch(BATCHSIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

test_ds = test_ds.map(normalize, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(128)
test_ds = test_ds.prefetch(AUTOTUNE)

model = Sequential(
    [
        tf.keras.Input((28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(train_ds, epochs=5)
print("Evaluate ----")
model.evaluate(test_ds)

model.save("saved_models/mnist.h5")
