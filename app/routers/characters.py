from fastapi import APIRouter, HTTPException

from app.schemas.character import CharacterCreate, CharacterRead


router = APIRouter(prefix="/characters", tags=["characters"])

characters = {}

@router.post("/", response_model=CharacterRead)
def create_character(payload: CharacterCreate):
    if payload.id in characters:
        raise HTTPException(status_code=400, detail="Character already exists!")
    characters[payload.id] = payload
    return payload

@router.get("/{character_id}", response_model=CharacterRead)
def read_character(character_id: str):
    if character_id not in characters:
        raise HTTPException(status_code=404, detail="Character not found")
    return characters[character_id]