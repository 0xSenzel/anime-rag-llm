from typing import List, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., example="How can I make the perfect cup of tea?")
    character: Optional[str] = Field(
        None,
        description="Optional anime character persona to role-play",
        example="Uncle Iroh"
    )
    use_rag: bool = Field(
        True,
        description="Whether to use Retrieval-Augmented Generation (RAG) with document context",
        example=True
    )
    user_id: str = Field(
        ...,
        description="Identifier for the user whose documents should be searched if RAG is enabled.",
        example="123"
    )
    rag_k: int = Field(
        5,
        description="Number of document chunks to retrieve for RAG.",
        example=5
    )

class ChatResponse(BaseModel):
    character: Optional[str]
    context: List[str]
    response: str