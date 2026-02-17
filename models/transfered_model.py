import os
from glob import glob

from models.model import Model
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.optimizers import Adam

from config import base_model_path


class TransferedModel(Model):
    @staticmethod
    def _resolve_base_model_path():
        if os.path.exists(base_model_path):
            return base_model_path

        candidates = sorted(glob('results/basic_model*.keras'))
        if candidates:
            return candidates[-1]

        raise FileNotFoundError(
            f"Unable to find a base model. Set config.base_model_path or place a basic model under results/."
        )

    def _define_model(self, input_shape, categories_count):
        model_path = self._resolve_base_model_path()
        base_model = models.load_model(model_path)

        # Remove original classification head and keep feature extractor
        feature_extractor = Sequential(base_model.layers[:-1], name='transfer_feature_extractor')
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
