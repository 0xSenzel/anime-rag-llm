import os
import uuid
import logging
from dotenv import load_dotenv
from fastapi import HTTPException
from supabase import create_client, Client
from typing import Optional

logger = logging.getLogger(__name__)

# Global client instance with type hint
_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    """
    Get or create a Supabase client instance.
    Implements singleton pattern to reuse the same client.
    
    Returns:
        Client: The Supabase client instance
        
    Raises:
        RuntimeError: If Supabase credentials are missing or connection fails
    """
    global _supabase_client
    if _supabase_client is None:
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            logger.error("Supabase credentials missing in environment variables.")
            raise RuntimeError("Supabase credentials missing.")
        
        try:
            _supabase_client = create_client(url, key)
            # Test the connection
            _supabase_client.auth.get_user()
            logger.info("Successfully connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise RuntimeError(f"Failed to connect to Supabase: {e}")
            
    return _supabase_client

def get_allowed_buckets() -> set[str]:
    """
    Retrieve allowed bucket names from environment variable (comma-separated).
    Raises an error if not set or empty.
    """
    buckets = os.getenv("SUPABASE_ALLOWED_BUCKETS")
    if not buckets:
        logger.error("SUPABASE_ALLOWED_BUCKETS environment variable is not set or empty.")
        raise RuntimeError("SUPABASE_ALLOWED_BUCKETS environment variable must be set with at least one bucket name.")
    return set(b.strip() for b in buckets.split(",") if b.strip())

def get_default_bucket() -> str:
    """
    Retrieve the default bucket name from the environment variable.
    Raises an error if not set or empty.
    Optionally, ensures the default bucket is in the allowed buckets.
    """
    bucket = os.getenv("SUPABASE_DEFAULT_BUCKET")
    if not bucket or not bucket.strip():
        logger.error("SUPABASE_DEFAULT_BUCKET environment variable is not set or is empty.")
        raise RuntimeError("SUPABASE_DEFAULT_BUCKET environment variable must be set to a valid bucket name.")
    bucket = bucket.strip()

    # Optional: ensure the default bucket is in the allowed buckets
    allowed_buckets = get_allowed_buckets()
    if bucket not in allowed_buckets:
        logger.error(f"Default bucket '{bucket}' is not in allowed buckets: {allowed_buckets}")
        raise RuntimeError(f"Default bucket '{bucket}' is not in allowed buckets: {allowed_buckets}")

    return bucket


async def upload_file_to_supabase(
        local_path: str, 
        original_filename: str, 
        bucket_name: str,
        folder: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
) -> str:
    """
    Uploads a file to Supabase Storage in the specified bucket and folder, generating a UUID-based key.
    Optionally includes user_id and conversation_id in the storage path for easier filtering.
    """
    supabase = get_supabase_client()
    allowed_buckets = get_allowed_buckets()
    bucket = bucket_name or get_default_bucket()
    if bucket not in allowed_buckets:
        logger.error(f"Bucket '{bucket}' is not in allowed buckets: {allowed_buckets}")
        raise HTTPException(status_code=400, detail=f"Bucket '{bucket}' is not allowed.")

    file_uuid = str(uuid.uuid4())
    ext = os.path.splitext(original_filename)[1]

    # Build the storage key with identifiers
    path_parts = []
    if folder:
        path_parts.append(folder.rstrip('/'))
    if user_id:
        path_parts.append(str(user_id))
    if conversation_id:
        path_parts.append(str(conversation_id))
    path_parts.append(f"{file_uuid}{ext}")
    storage_key = "/".join(path_parts)

    logger.info(f"Uploading to path '{storage_key}' in bucket '{bucket}'")

    try:
        with open(local_path, "rb") as f:
            res = supabase.storage.from_(bucket).upload(storage_key, f)
        logger.info(f"File uploaded to Supabase Storage at key: {storage_key} in bucket: {bucket}")
        return storage_key
    except Exception as e:
        logger.error(f"Exception during Supabase upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Supabase upload failed.")