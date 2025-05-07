import os
from dotenv import load_dotenv
import logging
from google import genai
from typing import Iterable, Optional, List, Dict, Any
from google.genai.types import GenerateContentResponse
import asyncio
import tiktoken
from google.api_core import exceptions as google_exceptions
from .vector_store import VectorStoreService

load_dotenv()  
logger = logging.getLogger(__name__)

# --- LlmService Class ---

class LlmService:
    """
    Handles interactions with the Google Generative AI models for text generation,
    streaming responses, summarization, and tokenization estimation.
    """
    def __init__(
        self,
        vector_store_svc: VectorStoreService,
        api_key: str,
        default_model: str,
        summary_model: str,
        tokenizer_encoding: str = "cl100k_base",
    ):
        """
        Initializes the LlmService.

        Args:
            vector_store_svc: An initialized VectorStoreService instance for vector search and RAG.
            api_key: Google API Key for authenticating with the Google GenAI service.
            default_model_name: Default model name for text generation.
            summary_model_name: Model name to use for summarization tasks.
            tokenizer_encoding: The tiktoken encoding to use for token counting estimates (default: "cl100k_base").
        """

        logger.info(f"Initializing LlmService with default_model='{default_model}', summary_model='{summary_model}', tokenizer_encoding='{tokenizer_encoding}'")

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Client: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Google GenAI Client: {e}") from e

        self.tokenizer = self._load_tokenizer(tokenizer_encoding)
        if not self.tokenizer:
             logger.warning("Tokenizer not available. Token counting features will be disabled.")
        self.vector_store_svc = vector_store_svc
        if not self.vector_store_svc:
            logger.error("VectorStoreService instance not provided. LlmService requires a valid vector store for RAG features.")
            raise ValueError("LlmService requires a valid VectorStoreService instance.")


    def _load_tokenizer(self, encoding_name: str) -> Optional[Any]:
        """Loads the tiktoken tokenizer."""
        if tiktoken:
            try:
                tokenizer = tiktoken.get_encoding(encoding_name)
                logger.info(f"Successfully loaded tiktoken tokenizer with encoding: {encoding_name}")
                return tokenizer
            except Exception as e:
                logger.error(f"Failed to load tiktoken tokenizer encoding '{encoding_name}': {e}", exc_info=True)
                return None
        else:
            logger.warning("tiktoken library not found. Install with `pip install tiktoken` for token counting.")
            return None

    def count_tokens(self, text: str) -> int:
        """Estimates the number of tokens in a string using the loaded tokenizer."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.error(f"Error encoding text for token count: {e}", exc_info=True)
                return 0 # Return 0 or raise? 0 is safer.
        return 0 # No tokenizer available

    def _prepare_prompt_and_context(
        self,
        user_query: str,
        context_docs: Optional[List[Dict]] = None, # Changed from List[Document] to List[Dict]
        character: Optional[str] = None,
        context_source_type: str = "documents" # e.g., 'documents', 'messages'
        ) -> str:
        """Helper to construct the final prompt string including context and instructions."""
        context_str = ""
        if context_docs:
            context_str = f"\n\nContext from relevant {context_source_type}:\n"
            for doc in context_docs:
                metadata = doc.get("metadata", {})
                content = doc.get("content", "[Missing Content]")
                # Try getting source info, common keys might be 'source' or 'document_id'
                source_info = metadata.get("source", metadata.get("document_id", "N/A"))
                context_str += f"- Source: {source_info}\n"
                context_str += f"  Content: {content}\n"
            context_str += "\n"

        # Base instruction depends on whether context is present
        base_instruction = ""
        if context_docs:
             base_instruction = (
                 f"Answer the following query based *only* on the provided context from {context_source_type}. "
                 "If the context doesn't contain the answer, clearly state that based on the provided information. "
             )
        else:
            base_instruction = "Answer the following query. "

        # Character instruction overrides/prepends
        char_instruction = ""
        if character:
            char_instruction = (
                f"You are {character}, a character from an anime. "
                f"Respond while staying in character as {character}. "
                f"Do not break character. Do not mention that you are an AI model. "
            )
            # Combine: Character persona comes first
            full_instruction = f"{char_instruction}{base_instruction}"
        else:
            full_instruction = base_instruction

        final_prompt = f"{context_str}{full_instruction}Query: \"{user_query}\""
        return final_prompt

    def _generate_sync_stream(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> Iterable[GenerateContentResponse]:
        """Synchronously calls the Google API stream method."""
        effective_model = model or self.default_model
        logger.debug(f"Generating sync stream with model={effective_model}")
        logger.debug(f"Full Prompt for Sync Stream:\n{prompt}")
        try:
            response: Iterable[GenerateContentResponse] = self.client.models.generate_content_stream(
                model=effective_model,
                contents=prompt,
                # Add safety_settings, generation_config if needed
            )
            return response
        except google_exceptions.GoogleAPIError as api_err:
             logger.error(f"Google API Error during streaming generation with {effective_model}: {api_err}", exc_info=True)
             # Re-raise a more specific or generic error as needed
             raise ConnectionError(f"LLM API Error: {api_err}") from api_err
        except Exception as e:
            logger.error(f"Unexpected error generating content stream with {effective_model}: {e}", exc_info=True)
            raise


    async def stream_llm_responses(
        self,
        user_query: str,
        user_id: str,
        character: Optional[str] = None,
        use_rag: bool = False,
        rag_k: int = 5,
        model: Optional[str] = None,
    ):
        """
        Async generator yields text chunks for a query. Optionally uses RAG
        to retrieve document chunks relevant to the user_id.

        Args:
            user_query: The user's input query.
            user_id: The ID of the user, used for filtering RAG results.
            vector_store: An initialized instance of VectorStoreService.
            character: Optional character persona.
            use_rag: Flag to enable RAG using uploaded documents.
            rag_k: Number of document chunks to retrieve for RAG.
            model: LLM model name (defaults to instance default).
        Yields:
            str: Text chunks from the LLM response.
        """
        retrieved_context: Optional[List[Dict]] = None
        if use_rag:
            try:
                logger.info(f"Retrieving document chunks for RAG (user: {user_id}, k={rag_k})")
                # Use the method for searching document chunks
                retrieved_context = await self.vector_store_svc.search_relevant_messages(
                    conversation_id='',
                    query_text=user_query,
                    k=rag_k
                )
                if not retrieved_context:
                    logger.warning(f"RAG enabled for user {user_id}, but no relevant document chunks found for the query.")
                else:
                     logger.info(f"Found {len(retrieved_context)} relevant document chunks for RAG.")
            except Exception as e:
                logger.error(f"Error retrieving document chunks for RAG (user: {user_id}): {e}", exc_info=True)
                yield "[Error retrieving documents for RAG]"
                return # Stop generation if RAG retrieval fails

        # Prepare the final prompt using the helper
        final_prompt = self._prepare_prompt_and_context(
            user_query=user_query,
            context_docs=retrieved_context, # Pass dicts directly
            character=character,
            context_source_type="uploaded documents"
        )

        try:
            # Get the synchronous stream generator
            sync_iterable_response = self._generate_sync_stream(
                prompt=final_prompt,
                model=model # Uses instance default if None
            )

            # Consume the synchronous stream in a separate thread
            async for text_chunk in self._consume_stream_async(sync_iterable_response):
                yield text_chunk

        except ConnectionError as conn_err: # Catch specific LLM connection errors
             logger.error(f"LLM Connection Error during streaming: {conn_err}", exc_info=True)
             yield f"[LLM Stream Error: {conn_err}]"
        except Exception as e:
            logger.error(f"Error setting up or running LLM streaming: {e}", exc_info=True)
            yield "[STREAMING SETUP ERROR]"

    async def _consume_stream_async(self, sync_iterable: Iterable[GenerateContentResponse]):
        """Helper async generator to consume the sync stream from Google API."""
        try:
            for chunk in await asyncio.to_thread(list, sync_iterable): # Consume all in thread
                try:
                    text_chunk = ""
                    # Safer access, aligning with Google GenAI library structure
                    if hasattr(chunk, 'text'):
                        text_chunk = chunk.text
                    elif hasattr(chunk, 'candidates') and chunk.candidates:
                        content = getattr(chunk.candidates[0], 'content', None)
                        if content and hasattr(content, 'parts') and content.parts:
                            text_chunk = getattr(content.parts[0], 'text', "")

                    if text_chunk:
                        yield text_chunk
                        # Optional slight delay, consider removing if not needed
                        # await asyncio.sleep(0.01)
                except StopIteration:
                    # Expected when the sync stream finishes within the list conversion
                    break
                except Exception as proc_e:
                    logger.warning(f"Could not process text from chunk: {proc_e}. Chunk: {chunk}", exc_info=True)
                    yield "[Error processing chunk]"
        except Exception as outer_e:
             logger.error(f"Error consuming synchronous LLM stream in thread: {outer_e}", exc_info=True)
             yield "[Error consuming LLM stream]"


    async def generate_response_async(
        self,
        user_query: str,
        context_docs: Optional[List[Dict]] = None,
        character: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[str]:
        """
        Generates a non-streaming response asynchronously.

        Args:
             user_query: The user's input query.
             context_docs: Optional list of context document dictionaries.
             character: Optional character persona.
             model: LLM model name (defaults to instance default).

        Returns:
             The generated text response as a string, or None on failure.
        """
        effective_model = model or self.default_model
        final_prompt = self._prepare_prompt_and_context(
             user_query=user_query,
             context_docs=context_docs,
             character=character,
             context_source_type="documents" # Generic source type for non-stream
        )
        logger.info(f"Generating non-streaming response with model {effective_model}")
        logger.debug(f"Full Prompt for Non-Streaming:\n{final_prompt}")

        try:
            # Use asyncio.to_thread for the blocking SDK call
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=effective_model,
                contents=final_prompt,
                # Add safety_settings, generation_config if needed
            )

            # Extract text safely
            text_response = None
            if hasattr(response, 'text'):
                 text_response = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                 candidate = response.candidates[0]
                 if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                     text_response = getattr(candidate.content.parts[0], 'text', None)

            if text_response:
                 return text_response.strip()
            else:
                 logger.warning(f"LLM response did not contain expected text structure. Response: {response}")
                 return None

        except google_exceptions.GoogleAPIError as api_err:
            logger.error(f"Google API Error during non-streaming generation with {effective_model}: {api_err}", exc_info=True)
            return None # Or raise custom error
        except Exception as e:
            logger.error(f"Unexpected error during non-streaming generation with {effective_model}: {e}", exc_info=True)
            return None # Or raise


    async def summarize_text_async(self, text_to_summarize: str, model: Optional[str] = None) -> Optional[str]:
        """
        Generates a summary for the given text asynchronously using the specified model.

        Args:
            text_to_summarize: The long text content needing summarization.
            model: The specific model to use (defaults to instance summary model).

        Returns:
            The summary text string, or None on failure.
        """
        effective_model = model or self.summary_model
        logger.info(f"Generating summary using model {effective_model}...")
        summary_prompt = f"Provide a concise summary of the key points discussed in the following text:\n\n---\n{text_to_summarize}\n---\n\nSummary:"

        try:
            # Use the non-streaming method via to_thread
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=effective_model,
                contents=summary_prompt,
            )

            # Extract text safely (same logic as generate_response_async)
            summary = None
            if hasattr(response, 'text'):
                 summary = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                 candidate = response.candidates[0]
                 if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                     summary = getattr(candidate.content.parts[0], 'text', None)

            if summary:
                logger.info("Summary generation successful.")
                return summary.strip()
            else:
                logger.warning(f"LLM summary response did not contain expected text structure. Response: {response}")
                return None
        except google_exceptions.GoogleAPIError as api_err:
            logger.error(f"Google API Error during summarization with {effective_model}: {api_err}", exc_info=True)
            return None # Or raise custom error
        except Exception as e:
            logger.error(f"Error during LLM summarization call with {effective_model}: {e}", exc_info=True)
            return None