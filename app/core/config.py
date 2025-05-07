import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploaded_files")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "textembedding-gecko@001")
    RAG_RELEVANCE_THRESHOLD: float = 0.5 # Default threshold (adjust based on testing!)

settings = Settings()