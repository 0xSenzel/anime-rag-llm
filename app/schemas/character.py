from typing import List
from pydantic import BaseModel, Field


class CharacterBase(BaseModel):
    id: str = Field(..., example="naruto")
    name: str = Field(..., example="Naruto Uzumaki")
    bio: str = Field(..., example="A spirited ninja...")
    quotes: List[str] = Field(..., example=["Believe it!", "Never give up!"])


class CharacterCreate(CharacterBase):
    pass

class CharacterRead(CharacterBase):
    pass

class QueryRequest(BaseModel):
    query: str = Field(..., example="How can i get stronger?")

class RAGResponse(BaseModel):
    character_id: str
    character_name: str
    prompt: str
    response: str