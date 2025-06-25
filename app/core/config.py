import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

load_dotenv()

class Settings(BaseSettings):
    UPLOAD_DIR: str = "uploaded_files"
    EMBEDDING_MODEL: str = "textembedding-gecko@001"
    RAG_RELEVANCE_THRESHOLD: float = 0.5 # Default threshold (adjust based on testing!)
    
    # Auth settings
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Google OAuth settings
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    
    # Add all the fields you use in your .env or environment
    GOOGLE_API_KEY: str
    MODEL_NAME: str
    SUMMARY_MODEL_NAME: str
    PINECONE_API_KEY: str
    PINECONE_HOST: str
    PINECONE_INDEX_NAME: str
    PINECONE_METRIC: str
    PINECONE_USE_SERVERLESS: bool
    PINECONE_CLOUD: str
    PINECONE_REGION: str
    SUPABASE_DB_HOST: str
    SUPABASE_DB_PORT: int
    SUPABASE_DB_NAME: str
    SUPABASE_DB_USER: str
    SUPABASE_DB_PASSWORD: str
    SUPABASE_DB_SSL_MODE: str
    SUPABASE_S3_ACCESS_KEY_ID: str
    SUPABASE_S3_SECRET_ACCESS_KEY: str
    SUPABASE_S3_ENDPOINT: str
    SUPABASE_S3_REGION: str
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_ALLOWED_BUCKETS: str
    SUPABASE_DEFAULT_BUCKET: str

    class Config:
        env_file = ".env"

settings = Settings()