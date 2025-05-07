from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
import os
import shutil # Keep shutil for copyfileobj
import logging # Added for logging
from app.core.config import settings
from app.services.document_processor import load_and_split
from app.services.vector_store import VectorStoreService
from app.core.dependencies import get_vector_store_service

# Added logger
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...), # Added type hint and Form(...) to receive user_id from form data
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Handles file uploads:
    1. Validates file type (.pdf, .txt).
    2. Saves the file temporarily.
    3. Processes the file content using load_and_split.
    4. Adds the processed chunks to the vector store using VectorStoreService.
    5. Cleans up the temporary file.
    """
    # 1. Validate file type
    if not file.filename or not file.filename.endswith(('.pdf', '.txt')):
        logger.warning(f"Upload rejected: Unsupported file type '{file.filename}'")
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .txt are allowed.")

    # Define file_path outside the try block so it's accessible in finally
    file_path = None
    try:
        # Ensure upload directory exists
        try:
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create upload directory '{settings.UPLOAD_DIR}': {e}", exc_info=True)
            # Use 500 for server-side configuration issues
            raise HTTPException(status_code=500, detail="Server configuration error: Could not prepare upload location.")

        # Construct the full path for the temporary file
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        logger.info(f"Receiving file: '{file.filename}', saving temporarily to '{file_path}'")

        # 2. Save the uploaded file temporarily
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File '{file.filename}' saved successfully to temporary location.")
        except Exception as save_err:
             logger.error(f"Failed to save uploaded file '{file.filename}' to '{file_path}': {save_err}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Could not save uploaded file '{file.filename}'.")


        # 3. Process the saved file: Load & Split
        logger.info(f"Processing and splitting file '{file.filename}'...")
        try:
            chunks = load_and_split(file_path)
            logger.info(f"File '{file.filename}' split into {len(chunks)} chunks.")
        except ValueError as ve: # Catch specific errors like unsupported file type from load_and_split
            logger.warning(f"Value error during document splitting of '{file.filename}': {ve}", exc_info=True)
            # This indicates an issue with the file content/type after saving
            raise HTTPException(status_code=400, detail=f"Processing error for '{file.filename}': {ve}")
        except Exception as split_err:
            logger.error(f"Unexpected error splitting document '{file.filename}': {split_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process document content for '{file.filename}'.")

        # 4. Add processed chunks to Vector Store
        # This now calls the Pinecone version of add_documents
        logger.info(f"Adding {len(chunks)} chunks from '{file.filename}' to the vector store...")
        try:
            # Capture the boolean return value from the updated add_documents
            success = await vector_store.add_message_embedding(documents=chunks, user_id=user_id)

            if not success:
                logger.error(f"Failed to add document chunks from '{file.filename}' to the vector store. Check vector_store service logs for details.")
                # Raise specific error if adding to vector store failed
                # The vector_store function logs specifics, keep detail generic here.
                raise HTTPException(status_code=500, detail=f"Failed to index document '{file.filename}'.")
        except Exception as index_err:
            # Catch potential exceptions during the call to add_documents itself
            logger.error(f"Unexpected error calling add_documents for '{file.filename}': {index_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during document indexing for '{file.filename}'.")


        logger.info(f"Successfully processed and indexed '{file.filename}'.")
        # If everything succeeded up to this point
        return {"message": f"Successfully processed and indexed '{file.filename}'"}

    # Catch any HTTPException raised explicitly above
    except HTTPException as http_exc:
        raise http_exc
    # Catch any other unexpected errors not caught by specific blocks
    except Exception as e:
        logger.error(f"An unexpected error occurred in the upload endpoint for file '{file.filename or 'unknown'}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred during file upload.")

    finally:
        # 5. Clean up the temporary file
        # This block executes whether the try block succeeded or failed
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: '{file_path}'")
            except OSError as e:
                # Log an error if cleanup fails, but don't prevent response/error propagation
                logger.error(f"Error removing temporary file '{file_path}': {e}", exc_info=True)
