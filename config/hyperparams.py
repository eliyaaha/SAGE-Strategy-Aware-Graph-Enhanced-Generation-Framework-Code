import torch

# --- General Settings ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Paths ---
RAW_METADATA_PATH = "/data/annotation_set.csv"
BLOCKED_CONV_PATH = "/data/conversations.csv"
LEXICON_PATH = "/data/lexicon_dict_5.3.json"
PROCESSED_DATA_PATH = "./outputs/processed_data_with_lexicon.csv"
OUTPUT_DIR = "./models_checkpoints/"

# --- Model Names ---
BASE_LLM_MODEL = "google/gemma-3-12b-it" 
ENCODER_MODEL = "onlplab/alephbert-base"

# --- HGT Hyperparameters ---
TARGET_LABELS = {"דיבוב", "שיקוף", "מתן נקודה למחשבה"}
GNN_HIDDEN_DIM = 256
GNN_LR = 1e-3
GNN_WD = 1e-5
GNN_EPOCHS = 200
GNN_PATIENCE = 10

