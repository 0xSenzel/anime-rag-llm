from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os,logging

load_dotenv()

logger = logging.getLogger(__name__)

DB_HOST = os.getenv("SUPABASE_DB_HOST")
DB_PORT = os.getenv("SUPABASE_DB_PORT", 5432) # Default to Supabase pooler port
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")
DB_SSL_MODE = os.getenv("SUPABASE_DB_SSL_MODE", "require") # Default to require for Supabase

if not all([DB_HOST, DB_PASSWORD]):
    logger.error("Missing required Supabase database environment variables (SUPABASE_DB_HOST, SUPABASE_DB_PASSWORD).")
    # You might raise an exception here or exit depending on your app's startup requirements
    raise ValueError("Missing required Supabase environment variables.")

# Connection string: postgresql://<user>:<pass>@<host>:<port>/<db>
SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logger.info(f"Connecting to database: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")

connects_args = {}
if DB_SSL_MODE != 'disable':
    connects_args["sslmode"] = DB_SSL_MODE

try:
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args=connects_args,
        pool_size=10,   # Adjust pool size based on expected concurrency
        max_overflow=20 # Adjust based on expected peak load
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    logger.info("SQLAlchemy engine and session configured successfully.")

except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine or configure session: {e}", exc_info=True)
    raise

def get_db():
    """
    FastAPI dependency that provides a database session per request.
    Ensures the session is always closed, even if errors occur.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()