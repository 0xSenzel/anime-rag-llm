from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, List, Dict
from datetime import datetime

class CharacterBaseSchema(BaseModel):
    name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        example="Iroh",
        description="Unique character name. Should be alphanumeric, allowing spaces and apostrophes."
    )
    description: Optional[str] = Field(
        default=None,
        min_length=50,
        max_length=5000,
        example="Wise mentor renowned for compassion and unwavering belief in the potential for good in everyone, especially his nephew Zuko.",
        description="Detailed description of the character's personality, role, and significant traits."
    )
    backstory: Optional[str] = Field(
        default=None,
        min_length=100,
        max_length=5000,
        example="Son of Fire Lord Azulon, Iroh was a celebrated general before personal tragedy led him to a path of wisdom and peace.",
        description="The character's history and background."
    )
    motivations: Optional[str] = Field(
        default=None,
        min_length=50,
        max_length=1000,
        example="To guide Prince Zuko towards a path of honor and to help restore balance to the world.",
        description="The character's primary goals and driving forces."
    )
    aliases: Optional[List[str]] = Field(
        default=None,
        example=["General Iroh", "The Dragon of the West"],
        max_items=10,
        description="Max 10 alternative names or titles."
    )
    catchphrases: Optional[List[str]] = Field(
        default=None,
        example=["Come, let us have some tea...", "Pride is not the opposite of shame, but its source."],
        max_items=20,
        description="Character's famous or frequently used phrases."
    )
    sample_dialogues: Optional[List[Dict[str, str]]] = Field(
        default=None,
        example=[{"q": "Why do you smile so much?", "a": "It is how I choose to face the challenges of life."}],
        max_items=50,
        description="Example question and answer pairs demonstrating character's speech and personality."
    )
    knowledge_domains: Optional[List[str]] = Field(
        default=None,
        example=["firebending", "philosophy", "tea brewing", "Pai Sho strategy"],
        max_items=20,
        description="Areas of expertise or significant knowledge."
    )
    avatar_url: Optional[HttpUrl] = Field(
        default=None,
        example="https://static.wikia.nocookie.net/avatar/images/c/c1/Iroh_S3.png",
        description="A valid URL pointing to an image of the character."
    )

    @field_validator('name')
    @classmethod
    def validate_and_normalize_name(cls, v: Optional[str]):
        if v is None:
            return v
        temp_v = v.replace("'", "").replace(" ", "")
        if not temp_v or not temp_v.isalnum():
            raise ValueError("Name must effectively be alphanumeric when spaces and apostrophes are ignored.")
        return v.title()

    @field_validator('avatar_url')
    @classmethod
    def empty_url_to_none(cls, v):
        """Convert empty string to None for URL fields."""
        if v == "" or v is None:
            return None
        return v
        
    @field_validator('aliases', 'catchphrases', 'sample_dialogues', 'knowledge_domains')
    @classmethod
    def empty_list_to_none(cls, v):
        """Convert empty lists to None for optional list fields."""
        if v == [] or v == "":
            return None
        return v

    @field_validator('sample_dialogues')
    @classmethod
    def validate_dialogue_structure_and_length(cls, v: Optional[List[Dict[str, str]]]):
        if v:
            for dialogue_pair in v:
                if not isinstance(dialogue_pair, dict) or \
                   'q' not in dialogue_pair or \
                   'a' not in dialogue_pair:
                    raise ValueError("Each dialogue must be a dictionary containing 'q' and 'a' keys.")
                if not isinstance(dialogue_pair['q'], str) or \
                   not isinstance(dialogue_pair['a'], str):
                    raise ValueError("Dialogue 'q' and 'a' values must be strings.")
                if len(dialogue_pair['q']) > 200 or len(dialogue_pair['a']) > 500:
                    raise ValueError("Dialogue question (max 200 chars) or answer (max 500 chars) is too long.")
        return v

class CharacterMutationSchema(CharacterBaseSchema):
    pass

class CharacterCreateSchema(CharacterMutationSchema):
    name: str = Field(..., example="Zuko")
    description: str = Field(..., example="A troubled prince seeking to restore his honor...")
    backstory: str = Field(..., example="Exiled by his father, Fire Lord Ozai...")
    motivations: str = Field(..., example="To capture the Avatar and regain his place in the Fire Nation.")

class CharacterResponseSchema(CharacterBaseSchema):
    id: str
    name: str = Field(...)
    description: str = Field(...)
    backstory: str = Field(...)
    motivations: str = Field(...)

    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    query: str = Field(..., example="How can i get stronger?")

class RAGResponse(BaseModel):
    character_id: str
    character_name: str
    prompt: str
    response: str

class CharacterCreateUpdateResponseSchema(BaseModel):
    status: str
    message: str
    character_id: str