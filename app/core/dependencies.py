# Example: app/dependencies.py (or similar)
import os
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends
from app.services.vector_store import VectorStoreService, DEFAULT_VECTOR_DIMENSION
from app.services.llm_service import LlmService
from app.services.conversation import ConversationService
from app.database import get_db

# --- Caching Instances ---
# Use global variables for simple instance caching during app lifetime
_vector_store_service_instance = None
_llm_service_instance = None
# --- End Caching Instances ---

def get_vector_store_service() -> VectorStoreService:
    """
    Dependency function to get the VectorStoreService instance.
    Initializes it on first call using environment variables.
    """
    global _vector_store_service_instance
    if _vector_store_service_instance is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        # Add other necessary env vars based on your VectorStoreService __init__
        cloud = os.getenv("PINECONE_CLOUD")
        region = os.getenv("PINECONE_REGION")
        # environment = os.getenv("PINECONE_ENVIRONMENT") # If using pod-based

        if not api_key or not index_name:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pinecone configuration is missing in environment variables."
            )

        try:
            # Pass necessary config from env vars
            # Adjust based on whether you use serverless or pod-based
            _vector_store_service_instance = VectorStoreService(
                api_key=api_key,
                index_name=index_name,
                dimension=DEFAULT_VECTOR_DIMENSION, # Or get from env if needed
                use_serverless=True, # Or get from env
                cloud=cloud,
                region=region,
                # environment=environment # If pod-based
                # embedding_model=your_globally_managed_embedding_model # Optional: Inject model
            )
        except Exception as e:
             # Log the error appropriately
             print(f"FATAL: Could not initialize VectorStoreService: {e}")
             raise HTTPException(
                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail=f"Could not connect to vector store service: {e}"
             )

    return _vector_store_service_instance

def get_llm_service() -> LlmService:
    """
    Dependency function to get the LlmService instance.
    Initializes it on first call using environment variables implicitly
    handled by the LlmService class constructor.
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        try:
            # LlmService constructor handles reading env vars like GOOGLE_API_KEY
            # Pass constructor args here if you want to override env vars or defaults
            _llm_service_instance = LlmService()
        except (ValueError, ConnectionError) as e: # Catch specific init errors
             print(f"FATAL: Could not initialize LlmService: {e}") # Use logger in production
             raise HTTPException(
                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail=f"Could not initialize LLM service: {e}"
             )
        except Exception as e: # Catch any other unexpected errors
             print(f"FATAL: Unexpected error initializing LlmService: {e}") # Use logger
             raise HTTPException(
                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                 detail=f"Unexpected error initializing LLM service: {e}"
             )

    return _llm_service_instance

def get_conversation_service(
    db: Session = Depends(get_db),
    llm_service: LlmService = Depends(get_llm_service)
) -> ConversationService:
    """
    Dependency function to provide a ConversationService instance,
    with injected DB session and LlmService.
    """
    try:
        # Always create a new ConversationService per request to ensure fresh DB session
        return ConversationService(db=db, llm_service=llm_service)
    except Exception as e:
        # Log or print for debugging
        print(f"FATAL: Could not initialize ConversationService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not initialize ConversationService: {e}"
        )
        

