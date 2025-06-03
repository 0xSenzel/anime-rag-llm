from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class ChatRequest(BaseModel):
    query: str = Field(..., example="How can I make the perfect cup of tea?")
    user_id: str = Field(
        ...,
        description="Identifier for the user whose documents should be searched if RAG is enabled.",
        example="123"
    )
    conversation_id: Optional[uuid.UUID] = Field(
        None,
        description="Optional conversation ID to continue an existing conversation",
        example="5d3bfc49-6472-4c9a-966f-21afe51a8697"
    )
    character: Optional[str] = Field(
        None,
        description="Optional anime character persona to role-play",
        example="Uncle Iroh"
    )

class ChatResponse(BaseModel):
    character: Optional[str]
    context: List[str]
    response: str