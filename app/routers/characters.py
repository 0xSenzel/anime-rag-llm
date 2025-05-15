import logging
from typing import List, Dict
from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.character import (
    CharacterCreateSchema,
    CharacterMutationSchema,
    CharacterResponseSchema,
    CharacterCreateResponseSchema
)
from app.services.characters import CharacterService
from app.core.dependencies import get_character_service

logger = logging.getLogger(__name__)
 
router = APIRouter(prefix="/characters", tags=["characters"])


@router.post(
    "/",
    response_model=CharacterCreateResponseSchema,
    status_code=status.HTTP_201_CREATED,
)
async def create_new_character(
    character_data: CharacterCreateSchema,
    char_service: CharacterService = Depends(get_character_service)
):
    """
    Create a new character with the provided information.
    - **name**: Must be unique.
    - Other fields based on `CharacterCreateSchema`.
    """
    try:
        created_character = await char_service.create_character(character_data)
        return CharacterCreateResponseSchema(
            status="success",
            message=f"Character '{created_character.name}' created successfully",
            character_id=created_character.id
        )
    except HTTPException: # Re-raise HTTPExceptions directly from the service
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while creating the character."
        )

@router.get(
    "/{character_id}",
    response_model=CharacterResponseSchema,
    summary="Get a character by ID"
)
async def read_character_by_id(
    character_id: str,
    char_service: CharacterService = Depends(get_character_service)
):
    """
    Retrieve a single character by its unique ID.
    """
    try:
        return await char_service.get_character_by_id(character_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving character {character_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while retrieving the character."
        )

@router.get(
    "/",
    response_model=List[CharacterResponseSchema],
    summary="Get all characters"
)
async def read_all_characters(
    skip: int = 0,
    limit: int = 100,
    char_service: CharacterService = Depends(get_character_service)
):
    """
    Retrieve a list of all characters, with optional pagination.
    """
    try:
        return await char_service.get_all_characters(skip=skip, limit=limit)
    except Exception as e:
        logger.error(f"Unexpected error retrieving all characters: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while retrieving characters."
        )

@router.put(
    "/{character_id}",
    response_model=CharacterResponseSchema,
    summary="Update a character"
)
async def update_existing_character(
    character_id: str,
    character_update_data: CharacterMutationSchema, # Use CharacterMutationSchema for request body
    char_service: CharacterService = Depends(get_character_service)
):
    """
    Update an existing character's information.
    Only fields provided in the request body will be updated.
    """
    try:
        return await char_service.update_character(character_id, character_update_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating character {character_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while updating the character."
        )

@router.delete(
    "/{character_id}",
    response_model=Dict[str, str], # Standard message response
    summary="Delete a character"
)
async def remove_character(
    character_id: str,
    char_service: CharacterService = Depends(get_character_service)
):
    """
    Delete a character by its unique ID.
    """
    try:
        return await char_service.delete_character(character_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting character {character_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while deleting the character."
        )

# Optional: If you have a get_character_by_name in your service and want an endpoint for it
@router.get(
    "/name/{character_name}",
    response_model=CharacterResponseSchema,
    summary="Get a character by name (case-insensitive)"
)
async def read_character_by_name_endpoint( # Renamed to avoid conflict if you use name as a query param later
    character_name: str,
    char_service: CharacterService = Depends(get_character_service)
):
    """
    Retrieve a single character by its name (case-insensitive search).
    """
    try:
        return await char_service.get_character_by_name(character_name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving character by name '{character_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while retrieving the character by name."
        )