# config.py
# Paths and model settings shared by the app and the RAG pipeline.

from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"            # PDF knowledge base
INDEX_DIR = BASE_DIR / "index_faiss"    # FAISS index, rebuilt on every start
LOGO_PATH = BASE_DIR / "assets" / "aurahome.svg"
ENV_PATH = BASE_DIR / ".env"

# -----------------------------
# Models
# -----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 600

# -----------------------------
# Retrieval
# -----------------------------
K_RETRIEVE = 3          # fixed number of chunks sent to the model
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
CONTEXT_CLIP = 1200     # max characters kept from each retrieved chunk
HISTORY_WINDOW = 8      # previous messages sent with each question
