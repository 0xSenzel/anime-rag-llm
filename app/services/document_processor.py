import os
import shutil
import logging, uuid
from typing import List
from fastapi import UploadFile, HTTPException
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.vector_store import VectorStoreService
from app.services.storage import upload_file_to_supabase

logger = logging.getLogger(__name__)

__all__ = ['load_and_split', 'process_uploaded_file']

def load_and_split(file_path: str) -> List[Document]:
    # Load raw pages/text
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type.")
    
    docs = loader.load()
    # Split into 1,000‑char chunks with 200‑char overlap 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

async def process_uploaded_file(
    file: UploadFile,
    user_id: str,
    conversation_id: uuid.UUID,
    vector_store: VectorStoreService
) -> str:
    """
    Handles the full processing of an uploaded file:
    1. Validates file type.
    2. Saves the file temporarily.
    3. Processes the file content (load and split).
    4. Adds the processed chunks to the vector store.
    5. Cleans up the temporary file.
    Returns a success message or raises HTTPException on error.
    """
    # 1. Validate file type (already done in router, but good for service layer too)
    if not file.filename or not file.filename.endswith(('.pdf', '.txt')):
        logger.warning(f"Processing rejected: Unsupported file type '{file.filename}'")
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .txt are allowed.")

    file_path = None
    supabase_storage_key = None
    try:
        # Ensure upload directory exists
        try:
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create upload directory '{settings.UPLOAD_DIR}': {e}", exc_info=True)
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

        # --- Upload to Supabase Storage using new module ---
        try:
            supabase_storage_key = await upload_file_to_supabase(file_path, file.filename, "anime-rag", user_id=user_id, conversation_id=conversation_id)
            logger.info(f"File '{file.filename}' uploaded to Supabase Storage as '{supabase_storage_key}'")
        except Exception as supabase_err:
            logger.error(f"Supabase upload failed for '{file.filename}': {supabase_err}", exc_info=True)
            raise

        # 3. Process the saved file: Load & Split
        logger.info(f"Processing and splitting file '{file.filename}'...")
        try:
            chunks = load_and_split(file_path)
            logger.info(f"File '{file.filename}' split into {len(chunks)} chunks.")
        except ValueError as ve:
            logger.warning(f"Value error during document splitting of '{file.filename}': {ve}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Processing error for '{file.filename}': {ve}")
        except Exception as split_err:
            logger.error(f"Unexpected error splitting document '{file.filename}': {split_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process document content for '{file.filename}'.")

        # 4. Add processed chunks to Vector Store
        logger.info(f"Adding {len(chunks)} chunks from '{file.filename}' to the vector store for user '{user_id}'...")
        try:
            success = await vector_store.add_document_chunks(documents=chunks, user_id=user_id, conversation_id=conversation_id, namespace="documents")
            if not success:
                logger.error(f"Failed to add document chunks from '{file.filename}' to the vector store.")
                raise HTTPException(status_code=500, detail=f"Failed to index document '{file.filename}'.")
        except Exception as index_err:
            logger.error(f"Unexpected error calling add_message_embedding for '{file.filename}': {index_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during document indexing for '{file.filename}'.")

        logger.info(f"Successfully processed and indexed '{file.filename}'.")
        return f"Successfully processed and indexed '{file.filename}'"

    finally:
        # 5. Clean up the temporary file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Removed temporary file: '{file_path}'")
            except OSError as e:
                logger.error(f"Error removing temporary file '{file_path}': {e}", exc_info=True)