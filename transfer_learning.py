import requests
from tensorflow import keras
import tensorflow_hub as hub
import ssl

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


model_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
base_model = hub.KerasLayer(model_url, input_shape=(299, 299, 3), trainable=False)

model = keras.Sequential([base_model, keras.layers.Dense(128, activation="relu")])

print(model.summary())
