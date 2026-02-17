from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam


class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(categories_count, activation='softmax'),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
