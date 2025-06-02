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

__all__ = ['LlmService']

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
        conversation_id: Optional[uuid.UUID] = None,
        character: Optional[str] = None,
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
            model (Optional[str]): Name of LLM model to use. Defaults to instance default model.
        """
        full_llm_response_parts = []
        try:            
            if not conversation_id:
                conversation = self.conversation_svc.create_conversation(user_id)
                conversation_id = conversation.id
                logger.info(f"Created new conversation {conversation_id} for user {user_id}")
            else:
                try:
                    conversation = self.conversation_svc.get_conversation_by_id(user_id, conversation_id)
                    logger.debug(f"Using existing conversation {conversation_id}")
                except ValueError as e:
                    # If conversation_id is provided but not found for the user, log and create a new one
                    logger.warning(f"Conversation {conversation_id} not found for user {user_id}, creating new one: {e}")
                    conversation = self.conversation_svc.create_conversation(user_id)
                    conversation_id = conversation.id

            context_for_llm = await self.conversation_svc.get_context_for_llm(
                user_id=user_id,
                conversation_id=conversation_id,
                namespace=user_id,
                current_query=user_query
            )
            logger.debug(f"Assembled context for LLM: {len(context_for_llm)} parts")

            prompt = self._prepare_prompt_and_context(
                user_query=user_query,
                conversation_history=context_for_llm,
                rag_context_docs=None,
                character=character
            )

            # 4. Save the user message to DB and Pinecone (asynchronously)
            # Schedule the user message to be saved in the background.
            # The save_message function handles both DB persistence and Pinecone upserting.
            background_tasks.add_task(
                self.conversation_svc.save_message,
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                content=user_query
            )

            # 5. Stream response from LLM
            # The actual LLM call happens here, using the prepared prompt.
            sync_stream = self._generate_sync_stream(
                prompt=prompt,
                model=model
            )

            # 6. Process and yield stream chunks
            async for chunk in self._consume_stream_async(sync_stream):
                 yield chunk # Yield parts of the response as they come in
                 full_llm_response_parts.append(chunk) # Accumulate for saving

            # 7. Process full response (e.g., save assistant message, trigger summarization)
            full_response_content = "".join(full_llm_response_parts)
            if full_response_content:
                logger.info(f"Full LLM response received for conversation {conversation_id}. Content length: {len(full_response_content)}")
                # Save the assistant's response
                # Schedule the assistant message to be saved in the background.
                # This uses the save_message function which handles both DB and Pinecone.
                background_tasks.add_task(
                    self.conversation_svc.save_message,
                    conversation_id=conversation_id,
                    user_id=user_id, # Pass user_id to associate the message
                    role="assistant",
                    content=full_response_content
                )

                # Trigger background summarization if needed
                background_tasks.add_task(
                    self._background_summarize_if_needed,
                    conversation_id=conversation_id
                )
            else:
                logger.warning(f"Received empty LLM response for conversation {conversation_id}.")

        except ConnectionError as ce:
            logger.error(f"LLM Connection Error during stream for user {user_id}, convo {conversation_id}: {ce}", exc_info=True)
            yield f"An error occurred connecting to the AI model: {ce}"
        except Exception as e:
            logger.error(f"Unexpected error during stream for user {user_id}, convo {conversation_id}: {e}", exc_info=True)
            yield f"An unexpected error occurred: {e}"
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
            needs_summary, messages_to_summarize = self.conversation_svc.should_summarize(conversation_id)
            
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
                        await self.conversation_svc.save_summary(conversation_id, summary_text)
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