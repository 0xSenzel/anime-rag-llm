from pydantic import BaseModel, field_validator 
from fastapi import File, Form, UploadFile
import uuid

class UploadRequestForm(BaseModel):
    file: UploadFile
    user_id: str
    conversation_id: uuid.UUID

    class Config:
        arbitrary_types_allowed = True # To allow UploadFile type

    @field_validator('file')
    @classmethod
    def validate_file_type(cls, v: UploadFile):
        if not v.filename or not v.filename.endswith(('.pdf', '.txt')):
            raise ValueError("Unsupported file type. Only .pdf and .txt are allowed.")
        return v

    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v):
        if not isinstance(v, uuid.UUID):
            raise ValueError("conversation_id must be a valid UUID.")
        return v

async def get_validated_upload_form(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    conversation_id: str = Form(...)
) -> UploadRequestForm:
    """
    Dependency that creates and validates the UploadRequestForm.
    FastAPI will handle Pydantic's ValidationError and return a 422 response.

    This dependency is required when using Pydantic models for validating
    multipart/form-data that includes both File(...) uploads and Form(...) fields.
    FastAPI does not automatically populate a Pydantic model from mixed
    File and Form parameters directly. This function bridges that gap by:
    1. Receiving the individual File and Form parameters from FastAPI.
    2. Manually instantiating the Pydantic model (UploadRequestForm).
    3. This instantiation triggers Pydantic's own validation, including
       @field_validator methods defined in the UploadRequestForm model.
    """
    # Convert conversation_id to UUID (FastAPI will do this if annotated as uuid.UUID, but explicit conversion is safe)
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except Exception:
        raise ValueError("conversation_id must be a valid UUID string.")
    return UploadRequestForm(file=file, user_id=user_id, conversation_id=conversation_uuid)
