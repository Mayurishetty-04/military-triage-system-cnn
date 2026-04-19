import tensorflow as tf
try:
    print(f"TF Version: {tf.__version__}")
    import tensorflow.keras
    print("Successfully imported tensorflow.keras")
except Exception as e:
    print(f"Failed to import tensorflow.keras: {e}")

try:
    import keras
    print(f"Keras version: {keras.__version__}")
except Exception as e:
    print(f"Failed to import keras: {e}")
