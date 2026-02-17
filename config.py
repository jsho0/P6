categories = ['neutral', 'happy', 'surprise']

train_directory = 'train'
test_directory = 'test'

# Optional transfer-learning dataset directories.
# If these directories do not exist, code falls back to train_directory/test_directory.
transfer_train_directory = 'transfer_train'
transfer_test_directory = 'transfer_test'

train_size = 5000
original_image_size = (48, 48)
image_size = (150, 150)
batch_size = 128
validation_split = 0.2

# Path to a trained base model used for webcam inference and transfer learning.
# If missing, code will attempt to auto-discover the latest basic model in results/.
base_model_path = 'results/basic_model.keras'

BOARD_SIZE = 3
