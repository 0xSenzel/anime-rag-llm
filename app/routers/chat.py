from fastapi import APIRouter, HTTPException, Depends
from app.services.llm_service import LlmService
from app.services.vector_store import VectorStoreService
import logging
from app.schemas.chat import ChatRequest, ChatResponse
from app.core.dependencies import get_llm_service, get_vector_store_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/stream", response_model=ChatResponse, summary="Chat with RAG + Gemini")
async def chat_endpoint(
    request: ChatRequest,
    llm_service: LlmService = Depends(get_llm_service),
    vector_store_service: VectorStoreService = Depends(get_vector_store_service)
):
    """
    Handles chat requests and returns a ChatResponse with the LLM-generated response.
    """
    try:
        logger.info(f"Received chat request: '{request}'")
        answer = ""
        async for chunk in llm_service.stream_llm_responses(
            user_query=request.query,
            user_id=request.user_id,
            vector_store=vector_store_service,
            character=request.character,
            use_rag=request.use_rag,
            rag_k=request.rag_k
        ):
            answer += chunk

        return ChatResponse(
            character=request.character,
            context=[],
            response=answer,
        )
    except Exception as e:
        logger.exception("Error during LLM response generation:")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

