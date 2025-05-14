# Example: app/dependencies.py (or similar)
import os
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Depends
from app.services.vector_store import VectorStoreService, DEFAULT_VECTOR_DIMENSION
from app.services.llm_service import LlmService
from app.services.conversation import ConversationService
from app.database import get_db
from app.utils.tokenizer_service import TokenizerService
from app.services.characters import CharacterService

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

def get_tokenizer_service() -> TokenizerService:
    """
    Dependency function to get the TokenizerService instance.
    Initializes it on first call using environment variables.
    """
    try:
        return TokenizerService()
    except Exception as e:
        print(f"FATAL: Could not initialize TokenizerService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not initialize tokenizer service: {e}"
        )

def get_conversation_service(
    db: Session = Depends(get_db),
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
    tokenizer_svc: TokenizerService = Depends(get_tokenizer_service)
) -> ConversationService:
    """
    Dependency function to provide a ConversationService instance,
    with injected DB session and LlmService.
    """
    try:
        # Always create a new ConversationService per request to ensure fresh DB session
        return ConversationService(db=db, vector_store_svc=vector_store_svc, tokenizer_svc=tokenizer_svc)
    except Exception as e:
        # Log or print for debugging
        print(f"FATAL: Could not initialize ConversationService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not initialize ConversationService: {e}"
        )

def get_llm_service(
    vector_store_svc: VectorStoreService = Depends(get_vector_store_service),
    conversation_svc: ConversationService = Depends(get_conversation_service)    
) -> LlmService:
    """
    Provides a singleton instance of LlmService for FastAPI dependency injection.

    This function initializes the LlmService on first call using environment variables
    for configuration and the provided VectorStoreService and ConversationService instances.
    It ensures all required environment variables are present and raises an HTTPException
    if any are missing.

    Args:
        vector_store_svc (VectorStoreService): The vector store service instance for embeddings
                                              and similarity search.
        conversation_svc (ConversationService): The conversation service instance for managing
                                              chat history and context.

    Returns:
        LlmService: The singleton instance of the language model service.

    Raises:
        HTTPException: If required environment variables or dependencies are missing,
                      or if initialization fails.
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        default_model = os.getenv("MODEL_NAME")
        summary_model = os.getenv("SUMMARY_MODEL_NAME")

        if not api_key or not default_model or not summary_model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM configuration is missing in environment variables."
            )
        
        try:
            _llm_service_instance = LlmService(
                vector_store_svc=vector_store_svc,
                conversation_svc=conversation_svc,
                api_key=api_key,
                default_model=default_model,
                summary_model=summary_model
            )
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

def get_character_service(db: Session = Depends(get_db)) -> CharacterService:
    """
    Provides an instance of CharacterService for FastAPI dependency injection.
    
    This function creates a new CharacterService instance for each request,
    ensuring a fresh database session is used.
    
    Args:
        db (Session): SQLAlchemy database session from the get_db dependency.
        
    Returns:
        CharacterService: A new instance of the character service.
        
    Raises:
        HTTPException: If initialization fails.
    """
    try:
        return CharacterService(db=db)
    except Exception as e:
        print(f"FATAL: Could not initialize CharacterService: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not initialize character service: {e}"
        )

        

