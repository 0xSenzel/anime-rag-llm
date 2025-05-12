from fastapi import APIRouter, HTTPException, Depends
import logging

from app.services.document_processor import process_uploaded_file
from app.services.vector_store import VectorStoreService
from app.core.dependencies import get_vector_store_service
from app.schemas.upload import UploadRequestForm, get_validated_upload_form


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("/")
async def upload_file(
    form_data: UploadRequestForm = Depends(get_validated_upload_form),
    vector_store: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Handles file uploads by delegating to the document_processor service.
    File type validation is now handled by the UploadRequestForm Pydantic model.
    """
    try:
        message = await process_uploaded_file(
            file=form_data.file,
            user_id=form_data.user_id,
            vector_store=vector_store
        )
        return {"message": message}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors from the service call or router itself
        # Use form_data.file.filename if file object might be None or filename missing
        filename_for_log = form_data.file.filename if form_data and form_data.file else "unknown"
        logger.error(f"An unexpected error occurred in the upload endpoint for file '{filename_for_log}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred during file upload.")
