# Use TFDS image dataset with only train split, but use 75% split.

import tensorflow as tf
import tensorflow_datasets as tfds


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") >= 0.98:  # Experiment with changing this value
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

# this is used if there is only train split
(train_ds, test_ds), info_ds = tfds.load(
    "citrus_leaves",
    as_supervised=True,
    shuffle_files=True,
    split=["train[:75%]", "train[75%:]"],
    with_info=True,
)


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

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(256, 256, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(train_ds, epochs=25, callbacks=[callbacks], validation_data=test_ds)

# model.evaluate(test_ds)
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

model.save("saved_models/citrus-with-test.h5")
