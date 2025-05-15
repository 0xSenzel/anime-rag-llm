import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func
from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from app.schemas.character import (
    CharacterCreateSchema,
    CharacterMutationSchema,
    CharacterResponseSchema,
    CharacterBaseSchema,
)
from app.models.characters import Character

logger = logging.getLogger(__name__)

__all__ = ["CharacterService", "CharacterBaseSchema"]

class CharacterService:
    def __init__(self, db: Session):
        """
        Initializes the CharacterService with a database session.

        Args:
            db (Session): The SQLAlchemy database session.
        """
        self.db = db
        logger.debug(f"CharacterService initialized with db session: {db}")

    async def create_character(
        self, character_data: CharacterCreateSchema
    ) -> CharacterResponseSchema:
        """
        Creates a new character in the database.
        
        Args:
            character_data: The character data to create
            
        Returns:
            CharacterResponseSchema: The created character
            
        Raises:
            HTTPException: If character with same name exists or database error occurs
        """
        logger.info(f"Attempting to create character: {character_data.name}")
        
        try:
            # Check for existing character
            existing_character = (
                self.db.query(Character)
                .filter(sql_func.lower(Character.name) == sql_func.lower(character_data.name))
                .first()
            )
            if existing_character:
                logger.warning(
                    f"Character creation conflict: Name '{character_data.name}' already exists (ID: {existing_character.id})."
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Character with name '{character_data.name}' already exists.",
                )

            db_character = Character(**character_data.model_dump())
            
            self.db.add(db_character)
            self.db.commit()
            self.db.refresh(db_character)
            logger.info(f"Successfully created character '{db_character.name}' with ID: {db_character.id}")
            return CharacterResponseSchema.model_validate(db_character, from_attributes=True)
            
        except HTTPException:
            # Re-raise HTTP exceptions (they're already formatted correctly)
            raise
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Database integrity error creating character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Database constraint violation: {str(e)}"
            )
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error creating character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error creating character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

    async def get_character_by_id(
        self, character_id: str 
    ) -> CharacterResponseSchema:
        """
        Retrieves a character by ID.
        
        Args:
            character_id: The ID of the character to retrieve
            
        Returns:
            CharacterResponseSchema: The retrieved character
            
        Raises:
            HTTPException: If character not found or database error occurs
        """
        logger.debug(f"Attempting to retrieve character by ID: {character_id}")
        
        try:
            # Assuming character_id is a string that needs to be cast or handled by the DB driver for UUID
            db_character = self.db.query(Character).filter(Character.id == character_id).first()

            if not db_character:
                logger.warning(f"Character with ID '{character_id}' not found.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Character with ID '{character_id}' not found.",
                )
            logger.debug(f"Successfully retrieved character '{db_character.name}' (ID: {character_id}).")
            return CharacterResponseSchema.model_validate(db_character, from_attributes=True)
            
        except HTTPException:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error retrieving character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

    async def get_all_characters(
        self, skip: int = 0, limit: int = 10
    ) -> List[CharacterResponseSchema]:
        """
        Retrieves all characters with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[CharacterResponseSchema]: List of characters
            
        Raises:
            HTTPException: If database error occurs
        """
        logger.info(f"Retrieving all characters with skip={skip}, limit={limit}.")
        
        try:
            db_characters = self.db.query(Character).offset(skip).limit(limit).all()
            response_schemas = [
                CharacterResponseSchema.model_validate(char, from_attributes=True) for char in db_characters
            ]
            logger.info(f"Retrieved {len(response_schemas)} characters.")
            return response_schemas
            
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving characters: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error retrieving characters: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

    async def update_character(
        self, character_id: str, character_update_data: CharacterMutationSchema
    ) -> CharacterResponseSchema:
        """
        Updates a character by ID.
        
        Args:
            character_id: The ID of the character to update
            character_update_data: The data to update
            
        Returns:
            CharacterResponseSchema: The updated character
            
        Raises:
            HTTPException: If character not found, name conflict, or database error occurs
        """
        logger.info(f"Attempting to update character with ID: {character_id}")
        
        try:
            db_character = self.db.query(Character).filter(Character.id == character_id).first()
            if not db_character:
                logger.warning(f"Character with ID '{character_id}' not found for update.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Character with ID '{character_id}' not found to update.",
                )

            update_data_dict = character_update_data.model_dump(exclude_unset=True)
            logger.debug(f"Update data for character {character_id}: {update_data_dict}")

            # Convert HttpUrl to string if present
            if 'avatar_url' in update_data_dict:
                update_data_dict['avatar_url'] = str(update_data_dict['avatar_url'])

            if "name" in update_data_dict and update_data_dict["name"].lower() != db_character.name.lower():
                logger.debug(f"Name change detected for character {character_id}. Checking for conflicts.")
                existing_character_with_new_name = (
                    self.db.query(Character)
                    .filter(sql_func.lower(Character.name) == sql_func.lower(update_data_dict["name"]))
                    .filter(Character.id != character_id)
                    .first()
                )
                if existing_character_with_new_name:
                    logger.warning(
                        f"Update conflict for character {character_id}: New name '{update_data_dict['name']}' already taken by character {existing_character_with_new_name.id}."
                    )
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail=f"Another character with name '{update_data_dict['name']}' already exists.",
                    )
            
            for key, value in update_data_dict.items():
                setattr(db_character, key, value)
            
            self.db.commit()
            self.db.refresh(db_character)
            logger.info(f"Successfully updated character '{db_character.name}' (ID: {character_id}).")
            return CharacterResponseSchema.from_orm(db_character)
            
        except HTTPException:
            raise
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Database integrity error updating character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Database constraint violation: {str(e)}"
            )
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error updating character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error updating character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

    async def delete_character(self, character_id: str) -> dict:
        """
        Deletes a character by ID.
        
        Args:
            character_id: The ID of the character to delete
            
        Returns:
            dict: Success message
            
        Raises:
            HTTPException: If character not found or database error occurs
        """
        logger.info(f"Attempting to delete character with ID: {character_id}")
        
        try:
            db_character = self.db.query(Character).filter(Character.id == character_id).first()
            if not db_character:
                logger.warning(f"Character with ID '{character_id}' not found for deletion.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Character with ID '{character_id}' not found to delete.",
                )
            
            self.db.delete(db_character)
            self.db.commit()
            logger.info(f"Successfully deleted character with ID: {character_id}.")
            return {"message": f"Character with ID '{character_id}' deleted successfully."}
            
        except HTTPException:
            raise
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error deleting character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error deleting character: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )

    async def get_character_by_name(
        self, name: str
    ) -> CharacterResponseSchema:
        """
        Retrieves a character by name.
        
        Args:
            name: The name of the character to retrieve
            
        Returns:
            CharacterResponseSchema: The retrieved character
            
        Raises:
            HTTPException: If character not found or database error occurs
        """
        logger.info(f"Attempting to retrieve character by name: {name}")
        
        try:
            db_character = (
                self.db.query(Character)
                .filter(sql_func.lower(Character.name) == sql_func.lower(name))
                .first()
            )
            if not db_character:
                logger.warning(f"Character with name '{name}' not found.")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Character with name '{name}' not found.",
                )
            logger.info(f"Successfully retrieved character '{db_character.name}' by name.")
            return CharacterResponseSchema.from_orm(db_character)
            
        except HTTPException:
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving character by name: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error retrieving character by name: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred"
            )