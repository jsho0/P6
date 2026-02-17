import os
from glob import glob

import numpy as np

from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import Adam

from config import base_model_path


class RandomModel(Model):
    @staticmethod
    def _resolve_base_model_path():
        if os.path.exists(base_model_path):
            return base_model_path

        candidates = sorted(glob('results/basic_model*.keras'))
        if candidates:
            return candidates[-1]

        raise FileNotFoundError(
            "Unable to find a base model. Set config.base_model_path or place a basic model under results/."
        )

    def _define_model(self, input_shape, categories_count):
        model_path = self._resolve_base_model_path()
        base_model = models.load_model(model_path)

        feature_extractor = Sequential(base_model.layers[:-1], name='random_feature_extractor')
        self._randomize_layers(feature_extractor)
        feature_extractor.trainable = False

        self.model = Sequential([
            layers.Input(shape=input_shape),
            feature_extractor,
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(categories_count, activation='softmax'),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights:
                continue

            randomized = []
            for w in weights:
                scale = max(float(np.std(w)), 1e-2)
                randomized.append(np.random.normal(loc=0.0, scale=scale, size=w.shape).astype(w.dtype))
            layer.set_weights(randomized)
