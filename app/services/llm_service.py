from dotenv import load_dotenv
import logging, uuid
from google import genai
from typing import Iterable, Optional, List, Dict
from google.genai.types import GenerateContentResponse
import asyncio
from google.api_core import exceptions as google_exceptions
from .vector_store import VectorStoreService
from app.services.conversation import ConversationService
from fastapi import BackgroundTasks

load_dotenv()  
logger = logging.getLogger(__name__)

class LlmService:
    """
    A service class that handles interactions with Google's Generative AI models.
    
    This service provides functionality for:
    - Text generation with context from conversations and documents
    - Streaming responses for real-time interaction
    - Text summarization for conversation history
    - Integration with RAG (Retrieval Augmented Generation)
    - Character-based responses for anime personas
    """
    def __init__(
        self,
        vector_store_svc: VectorStoreService,
        conversation_svc: ConversationService,
        api_key: str,
        default_model: str,
        summary_model: str,
    ):
        """
        Initializes the LlmService.

        Args:
            vector_store_svc (VectorStoreService): An initialized VectorStoreService instance for vector search and RAG.
            conversation_svc (ConversationService): An initialized ConversationService instance for managing conversations.
            api_key (str): Google API Key for authenticating with the Google GenAI service.
            default_model (str): Default model name for text generation.
            summary_model (str): Model name to use for summarization tasks.

        Raises:
            ConnectionError: If initialization of Google GenAI Client fails.
            ValueError: If vector_store_svc or conversation_svc is not provided.
        """
        self.api_key = api_key
        self.default_model = default_model
        self.summary_model = summary_model

        logger.info(f"Initializing LlmService with default_model='{self.default_model}', summary_model='{self.summary_model}'")

        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI Client: {e}", exc_info=True)
            raise ConnectionError(f"Failed to initialize Google GenAI Client: {e}") from e

        self.vector_store_svc = vector_store_svc
        if not self.vector_store_svc:
            logger.error("VectorStoreService instance not provided. LlmService requires a valid vector store for RAG features.")
            raise ValueError("LlmService requires a valid VectorStoreService instance.")
        self.conversation_svc = conversation_svc
        if not self.conversation_svc:
            logger.error("ConversationService instance not provided. LlmService requires a valid ConversationService instance.")
            raise ValueError("LlmService requires a valid ConversationService instance.")

    def _prepare_prompt_and_context(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict]] = None,
        rag_context_docs: Optional[List[Dict]] = None,
        character: Optional[str] = None,
    ) -> str:
        """
        Helper to construct the final prompt string including conversation history,
        RAG context, and character instructions.
        """
        prompt_parts = []

        # 1. Character Instruction (if any, comes first)
        if character:
            char_instruction = (
                f"You are {character}, a character from an anime. "
                f"Respond while staying in character as {character}. "
                f"Do not break character. Do not mention that you are an AI model. "
            )
            prompt_parts.append(char_instruction)

        # 2. Conversation History (if any)
        if conversation_history:
            history_str = "Previous conversation:\n"
            for msg in conversation_history:
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                history_str += f"{role}: {content}\n"
            prompt_parts.append(history_str.strip())

        # 3. RAG Context (if any)
        if rag_context_docs:
            rag_str = "Relevant information from documents:\n"
            for doc in rag_context_docs:
                metadata = doc.get("metadata", {})
                content = doc.get("content", "[Missing Content]")
                source_info = metadata.get("source", metadata.get("document_id", "N/A"))
                rag_str += f"- Source: {source_info}\n  Content: {content}\n"
            prompt_parts.append(rag_str.strip())

        # 4. Base Instruction (depends on context provided)
        base_instruction_parts = []
        if conversation_history and rag_context_docs:
            base_instruction_parts.append("Based on the previous conversation and the relevant documents provided,")
        elif conversation_history:
            base_instruction_parts.append("Based on the previous conversation,")
        elif rag_context_docs:
            base_instruction_parts.append("Based on the relevant documents provided,")

        if rag_context_docs: # Specific instruction for RAG
             base_instruction_parts.append(
                 "answer the following query. If the documents don't contain the answer, clearly state that based on the provided information."
             )
        else:
            base_instruction_parts.append("answer the following query.")
        
        if base_instruction_parts:
            prompt_parts.append(" ".join(base_instruction_parts))

        # 5. User Query
        prompt_parts.append(f"Query: \"{user_query}\"")

        final_prompt = "\n\n".join(filter(None, prompt_parts))
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
        background_tasks: BackgroundTasks, 
        conversation_id: uuid.UUID,
        character: Optional[str] = None,
        use_rag: bool = False,
        rag_k: int = 5,
        model: Optional[str] = None,
    ):
        """
        Async generator that streams text responses for a user query, handling conversation context,
        RAG (Retrieval Augmented Generation), message persistence, and background summarization.

        Args:
            user_query (str): The user's input query text.
            user_id (str): Unique identifier for the user making the query.
            background_tasks (BackgroundTasks): FastAPI BackgroundTasks instance for async operations.
            conversation_id (uuid.UUID): Unique identifier for the conversation thread.
            character (Optional[str]): Name of anime character persona to use for responses.
            use_rag (bool): Whether to enable RAG using uploaded documents. Defaults to False.
            rag_k (int): Number of relevant document chunks to retrieve for RAG. Defaults to 5.
            model (Optional[str]): Name of LLM model to use. Defaults to instance default model.
        """
        full_llm_response_parts = []
        try:
            background_tasks.add_task(self.conversation_svc.save_message, conversation_id, user_id, "user", user_query)


            # Get previous conversation context
            conversation_history_context: Optional[List[Dict]] = None
            try:
                conversation_history_context = await self.conversation_svc.get_context_for_llm(
                    conversation_id=conversation_id,
                    current_query_content=user_query # To potentially exclude it if already saved
                )
            except Exception as e:
                logger.error(f"Error retrieving conversation context for {conversation_id}: {e}", exc_info=True)
                yield "[Error retrieving conversation context]"
                return

            # Retrieve RAG context if enabled
            rag_context_docs: Optional[List[Dict]] = None
            if use_rag:
                try:
                    logger.info(f"Retrieving document chunks for RAG (user: {user_id}, k={rag_k})")
                    rag_context_docs = await self.vector_store_svc.search_relevant_messages(
                        conversation_id=conversation_id,
                        query_text=user_query,
                        k=rag_k
                    )
                    if not rag_context_docs:
                        logger.warning(f"RAG enabled, but no relevant document chunks found.")
                    else:
                        logger.info(f"Found {len(rag_context_docs)} relevant document chunks for RAG.")
                except Exception as e:
                    logger.error(f"Error retrieving RAG document chunks: {e}", exc_info=True)
                    yield "[Error retrieving documents for RAG]"
                    return

            # Prepare the final prompt
            final_prompt = self._prepare_prompt_and_context(
                user_query=user_query,
                conversation_history=conversation_history_context,
                rag_context_docs=rag_context_docs,
                character=character
            )

            # Stream LLM response
            sync_iterable_response = self._generate_sync_stream(
                prompt=final_prompt,
                model=model # Uses instance default if None
            )

            async for text_chunk in self._consume_stream_async(sync_iterable_response):
                yield text_chunk
                full_llm_response_parts.append(text_chunk)

            # Save LLM's full response (background)
            if conversation_id and full_llm_response_parts:
                full_response_text = "".join(full_llm_response_parts)
                if full_response_text.strip(): # Only save if there's content
                    background_tasks.add_task(
                        self.conversation_svc.save_message,
                        conversation_id,
                        user_id, # Or a system/bot ID
                        "assistant",
                        full_response_text
                    )
            
            # Perform summarization if needed (background)
            if conversation_id:
                background_tasks.add_task(
                    self._background_summarize_if_needed,
                    conversation_id,
                )

        except ConnectionError as conn_err:
            logger.error(f"LLM Connection Error during streaming: {conn_err}", exc_info=True)
            yield f"[LLM Stream Error: {conn_err}]"
        except Exception as e:
            logger.error(f"Error setting up or running LLM streaming: {e}", exc_info=True)
            yield "[STREAMING SETUP ERROR]"
        finally:
            # Ensure any pending background tasks related to this request are robustly handled
            # (FastAPI handles this for tasks added to BackgroundTasks instance)
            pass


    async def summarize_text_async(self, text_to_summarize: str, model: Optional[str] = None) -> Optional[str]:
        """
        Summarizes the given text using the configured summary model.
        """
        if not text_to_summarize.strip():
            logger.info("No text provided to summarize.")
            return None

        effective_model = model or self.summary_model
        # Simple prompt, can be made more sophisticated
        prompt = f"Please provide a concise summary of the following conversation or text:\n\n---\n{text_to_summarize}\n---\n\nSummary:"
        logger.info(f"Requesting summarization with model {effective_model}")
        logger.debug(f"Summarization prompt: {prompt}")

        try:
            # Use asyncio.to_thread for the blocking SDK call
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=effective_model,
                contents=prompt,
                # Consider adding safety_settings and generation_config if needed
            )

            summary_text = None
            # Safer access to response text, common for Google GenAI library
            if hasattr(response, 'text'):
                 summary_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                 candidate = response.candidates[0]
                 if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                     summary_text = getattr(candidate.content.parts[0], 'text', None)
            
            if summary_text:
                 logger.info("Successfully generated summary.")
                 return summary_text.strip()
            else:
                 logger.warning("LLM did not return text for summarization.")
                 return None
        except google_exceptions.GoogleAPIError as api_err:
            logger.error(f"Google API Error during summarization with {effective_model}: {api_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during text summarization with {effective_model}: {e}", exc_info=True)
            return None


    async def summarize_text_async(self, text_to_summarize: str, model: Optional[str] = None) -> Optional[str]:
        """
        Summarizes the given text using the configured summary model.
        """
        if not text_to_summarize.strip():
            logger.info("No text provided to summarize.")
            return None

        effective_model = model or self.summary_model
        # Simple prompt, can be made more sophisticated
        prompt = f"Please provide a concise summary of the following conversation or text:\n\n---\n{text_to_summarize}\n---\n\nSummary:"
        logger.info(f"Requesting summarization with model {effective_model}")
        logger.debug(f"Summarization prompt: {prompt}")

        try:
            # Use asyncio.to_thread for the blocking SDK call
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=effective_model,
                contents=prompt,
                # Consider adding safety_settings and generation_config if needed
            )

            summary_text = None
            # Safer access to response text, common for Google GenAI library
            if hasattr(response, 'text'):
                 summary_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                 candidate = response.candidates[0]
                 if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                     summary_text = getattr(candidate.content.parts[0], 'text', None)
            
            if summary_text:
                 logger.info("Successfully generated summary.")
                 return summary_text.strip()
            else:
                 logger.warning("LLM did not return text for summarization.")
                 return None
        except google_exceptions.GoogleAPIError as api_err:
            logger.error(f"Google API Error during summarization with {effective_model}: {api_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during text summarization with {effective_model}: {e}", exc_info=True)
            return None

    async def _background_summarize_if_needed(
        self,
        conversation_id: uuid.UUID,
    ):
        """Helper to run summarization in the background."""
        try:
            needs_summary, messages_to_summarize = await self.conversation_svc.should_summarize(conv_id_uuid)
            
            if needs_summary and messages_to_summarize:
                logger.info(f"Summarization triggered for conversation {conversation_id}")
                
                # Format messages (List[MessageModel]) into a single string for the LLM
                # Assuming MessageModel has .role and .content attributes
                formatted_text_to_summarize = "\n".join(
                    [f"{msg.role.capitalize()}: {msg.content}" for msg in messages_to_summarize]
                )
                
                if formatted_text_to_summarize.strip():
                    summary_text = await self.summarize_text_async(formatted_text_to_summarize)
                    if summary_text:
                        # save_summary should be an async method in ConversationService expecting UUID
                        await self.conversation_svc.save_summary(conv_id_uuid, summary_text)
                        logger.info(f"Summary saved for conversation {conversation_id}")
                    else:
                        logger.warning(f"Summarization did not produce text for conversation {conversation_id}")
                else:
                    logger.warning(f"No actual content in messages to summarize for conversation {conversation_id}")
            elif needs_summary:
                # This case means should_summarize returned True but no messages, which might indicate an issue
                # in should_summarize or an edge case (e.g., summary needed by token count but messages list was empty).
                logger.warning(f"Summarization flagged for {conversation_id}, but no messages were provided to summarize.")
            # else:
                # logger.debug(f"Summarization not needed for conversation {conversation_id}")

        except ValueError as ve: # Catch specific error for invalid UUID string
            logger.error(f"Invalid conversation_id format for summarization: '{conversation_id}'. Error: {ve}", exc_info=True)
        except Exception as e:
            logger.error(f"Error in background summarization for conv {conversation_id}: {e}", exc_info=True)


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