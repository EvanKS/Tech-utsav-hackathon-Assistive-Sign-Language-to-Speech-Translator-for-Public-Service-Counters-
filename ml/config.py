"""
SignBridge ML Configuration
"""
import os

# Dataset paths
D1_PATH = os.environ.get(
    "D1_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "Sign-Language-Digits-Dataset-master",
                 "Sign-Language-Digits-Dataset-master", "Dataset")
)
d2_default = os.path.join(os.path.dirname(__file__), "..", "..", "dataset vedio dataset")
if not os.path.exists(d2_default):
    d2_default = os.path.join(os.path.dirname(__file__), "..", "..", "dataset vedio")
D2_PATH = os.environ.get("D2_PATH", d2_default)
SIGNALPHASET_STATIC_PATH = os.environ.get(
    "SIGNALPHASET_STATIC_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "SignAlphaSet dataset",
                 "SignAlphaSet", "SignAlphaSet", "SignAlphaSet")
)
SIGNALPHASET_DYNAMIC_PATH = os.environ.get(
    "SIGNALPHASET_DYNAMIC_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "SignAlphaSet dataset",
                 "SignAlphaSet", "ASL_dynamic", "ASL_dynamic")
)

# Model output paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Training params
STATIC_EPOCHS = 60
STATIC_BATCH_SIZE = 32
TEMPORAL_EPOCHS = 80
TEMPORAL_BATCH_SIZE = 8
TEMPORAL_SEQUENCE_LENGTH = 32

# Landmark feature size: 21 landmarks * 3 coords = 63
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63
