import os

from keras.utils import image_dataset_from_directory
from config import (
    train_directory,
    test_directory,
    transfer_train_directory,
    transfer_test_directory,
    image_size,
    batch_size,
    validation_split,
)


def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset


def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset


def get_transfer_datasets():
    transfer_train = transfer_train_directory if os.path.isdir(transfer_train_directory) else train_directory
    transfer_test = transfer_test_directory if os.path.isdir(transfer_test_directory) else test_directory

    return _split_data(transfer_train, transfer_test, batch_size, validation_split)
