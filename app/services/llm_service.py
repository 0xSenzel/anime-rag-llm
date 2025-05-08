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
from app.services.conversation import ConversationService

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
        conversation_svc: ConversationService,
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
        self.api_key = api_key
        self.default_model = default_model
        self.summary_model = summary_model

        logger.info(f"Initializing LlmService with default_model='{self.default_model}', summary_model='{self.summary_model}', tokenizer_encoding='{tokenizer_encoding}'")

        try:
            self.client = genai.Client(api_key=self.api_key)
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
        self.conversation_svc = conversation_svc
        if not self.conversation_svc:
            logger.error("ConversationService instance not provided. LlmService requires a valid ConversationService instance.")
            raise ValueError("LlmService requires a valid ConversationService instance.")


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


    async def _background_summarize_if_needed(
        self,
        conversation_id: str, # Expecting string UUID from caller
    ):
        """Helper to run summarization in the background."""
        try:
            conv_id_uuid = uuid.UUID(conversation_id) # Convert string to UUID for service calls
            
            # should_summarize now returns: Tuple[bool, Optional[List[MessageModel]]]
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

    async def stream_llm_responses(
        self,
        user_query: str,
        user_id: str,
        conversation_service: "ConversationService", # Added
        background_tasks: "BackgroundTasks",       # Added
        conversation_id: Optional[str] = None,     # Added, assuming str or uuid.UUID
        character: Optional[str] = None,
        use_rag: bool = False,
        rag_k: int = 5,
        model: Optional[str] = None,
    ):
        """
        Async generator yields text chunks for a query. Handles conversation context,
        RAG, message saving, and background summarization.

        Args:
            user_query: The user's input query.
            user_id: The ID of the user.
            conversation_service: Initialized instance of ConversationService.
            background_tasks: FastAPI BackgroundTasks instance.
            conversation_id: Optional ID of the existing conversation.
            character: Optional character persona.
            use_rag: Flag to enable RAG using uploaded documents.
            rag_k: Number of document chunks to retrieve for RAG.
            model: LLM model name (defaults to instance default).
        Yields:
            str: Text chunks from the LLM response.
        """
        full_llm_response_parts = []
        try:
            # 1. Get or create conversation (assuming ConversationService handles this)
            #    And save user message in the background
            #    This part assumes ConversationService.handle_new_message or similar
            #    can create/get conversation and save the user message.
            #    For simplicity, let's assume conversation_id is managed and user message is saved
            #    before calling this, or ConversationService has a method to do it.
            #    Let's refine this: ConversationService should provide the conversation object.

            # For now, let's assume conversation_id is either provided or a new one is implicitly handled
            # by conversation_service when messages are saved or context is fetched.
            # A more robust approach would be:
            # conv_obj = await conversation_service.get_or_create_conversation(user_id, conversation_id, character)
            # current_conversation_id = conv_obj.id
            # background_tasks.add_task(conversation_service.save_message, current_conversation_id, user_id, "user", user_query)
            
            # Simplified: Assuming conversation_id is managed by the caller or a previous step.
            # If not, it needs to be created/retrieved here.
            # For this example, we'll assume `conversation_id` is valid if provided,
            # and `ConversationService` methods can handle `None` to create a new one if necessary.

            # Save user query (background)
            # This requires a valid conversation_id. If it's a new chat, it needs to be created first.
            # Let's assume `conversation_service.handle_user_message` does this and returns conv_id
            # For now, we'll proceed assuming conversation_id is available or handled by `save_message`
            if conversation_id: # Or handle creation if None
                 background_tasks.add_task(conversation_service.save_message, conversation_id, user_id, "user", user_query)


            # 2. Get previous conversation context
            conversation_history_context: Optional[List[Dict]] = None
            if conversation_id:
                try:
                    # Assuming get_context_for_llm is async
                    conversation_history_context = await conversation_service.get_context_for_llm(
                        conversation_id=conversation_id,
                        current_query_content=user_query # To potentially exclude it if already saved
                    )
                except Exception as e:
                    logger.error(f"Error retrieving conversation context for {conversation_id}: {e}", exc_info=True)
                    # Decide if to yield error or proceed without history

            # 3. Retrieve RAG context if enabled
            rag_context_docs: Optional[List[Dict]] = None
            if use_rag:
                try:
                    logger.info(f"Retrieving document chunks for RAG (user: {user_id}, k={rag_k})")
                    rag_context_docs = await self.vector_store_svc.search_relevant_messages(
                        conversation_id=conversation_id or '', # Use actual conv_id or default
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

            # 4. Prepare the final prompt
            final_prompt = self._prepare_prompt_and_context(
                user_query=user_query,
                conversation_history=conversation_history_context,
                rag_context_docs=rag_context_docs,
                character=character
            )

            # 5. Stream LLM response
            sync_iterable_response = self._generate_sync_stream(
                prompt=final_prompt,
                model=model # Uses instance default if None
            )

            async for text_chunk in self._consume_stream_async(sync_iterable_response):
                yield text_chunk
                full_llm_response_parts.append(text_chunk)

            # 6. Save LLM's full response (background)
            if conversation_id and full_llm_response_parts:
                full_response_text = "".join(full_llm_response_parts)
                if full_response_text.strip(): # Only save if there's content
                    background_tasks.add_task(
                        conversation_service.save_message,
                        conversation_id,
                        user_id, # Or a system/bot ID
                        "assistant",
                        full_response_text
                    )
            
            # 7. Perform summarization if needed (background)
            if conversation_id:
                # Pass self (LlmService instance) to the background task if it needs to call summarize_text_async
                background_tasks.add_task(
                    self._background_summarize_if_needed,
                    conversation_service,
                    conversation_id,
                    background_tasks # Pass background_tasks again if the helper itself adds more tasks (not typical)
                                     # Or just the services it needs.
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
        conversation_id: str, # Expecting string UUID from caller
    ):
        """Helper to run summarization in the background."""
        try:
            conv_id_uuid = uuid.UUID(conversation_id) # Convert string to UUID for service calls
            
            # should_summarize now returns: Tuple[bool, Optional[List[MessageModel]]]
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

    async def stream_llm_responses(
        self,
        user_query: str,
        user_id: str,
        conversation_service: "ConversationService", # Added
        background_tasks: "BackgroundTasks",       # Added
        conversation_id: Optional[str] = None,     # Added, assuming str or uuid.UUID
        character: Optional[str] = None,
        use_rag: bool = False,
        rag_k: int = 5,
        model: Optional[str] = None,
    ):
        """
        Async generator yields text chunks for a query. Handles conversation context,
        RAG, message saving, and background summarization.

        Args:
            user_query: The user's input query.
            user_id: The ID of the user.
            conversation_service: Initialized instance of ConversationService.
            background_tasks: FastAPI BackgroundTasks instance.
            conversation_id: Optional ID of the existing conversation.
            character: Optional character persona.
            use_rag: Flag to enable RAG using uploaded documents.
            rag_k: Number of document chunks to retrieve for RAG.
            model: LLM model name (defaults to instance default).
        Yields:
            str: Text chunks from the LLM response.
        """
        full_llm_response_parts = []
        try:
            # 1. Get or create conversation (assuming ConversationService handles this)
            #    And save user message in the background
            #    This part assumes ConversationService.handle_new_message or similar
            #    can create/get conversation and save the user message.
            #    For simplicity, let's assume conversation_id is managed and user message is saved
            #    before calling this, or ConversationService has a method to do it.
            #    Let's refine this: ConversationService should provide the conversation object.

            # For now, let's assume conversation_id is either provided or a new one is implicitly handled
            # by conversation_service when messages are saved or context is fetched.
            # A more robust approach would be:
            # conv_obj = await conversation_service.get_or_create_conversation(user_id, conversation_id, character)
            # current_conversation_id = conv_obj.id
            # background_tasks.add_task(conversation_service.save_message, current_conversation_id, user_id, "user", user_query)
            
            # Simplified: Assuming conversation_id is managed by the caller or a previous step.
            # If not, it needs to be created/retrieved here.
            # For this example, we'll assume `conversation_id` is valid if provided,
            # and `ConversationService` methods can handle `None` to create a new one if necessary.

            # Save user query (background)
            # This requires a valid conversation_id. If it's a new chat, it needs to be created first.
            # Let's assume `conversation_service.handle_user_message` does this and returns conv_id
            # For now, we'll proceed assuming conversation_id is available or handled by `save_message`
            if conversation_id: # Or handle creation if None
                 background_tasks.add_task(conversation_service.save_message, conversation_id, user_id, "user", user_query)


            # 2. Get previous conversation context
            conversation_history_context: Optional[List[Dict]] = None
            if conversation_id:
                try:
                    # Assuming get_context_for_llm is async
                    conversation_history_context = await conversation_service.get_context_for_llm(
                        conversation_id=conversation_id,
                        current_query_content=user_query # To potentially exclude it if already saved
                    )
                except Exception as e:
                    logger.error(f"Error retrieving conversation context for {conversation_id}: {e}", exc_info=True)
                    # Decide if to yield error or proceed without history

            # 3. Retrieve RAG context if enabled
            rag_context_docs: Optional[List[Dict]] = None
            if use_rag:
                try:
                    logger.info(f"Retrieving document chunks for RAG (user: {user_id}, k={rag_k})")
                    rag_context_docs = await self.vector_store_svc.search_relevant_messages(
                        conversation_id=conversation_id or '', # Use actual conv_id or default
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

            # 4. Prepare the final prompt
            final_prompt = self._prepare_prompt_and_context(
                user_query=user_query,
                conversation_history=conversation_history_context,
                rag_context_docs=rag_context_docs,
                character=character
            )

            # 5. Stream LLM response
            sync_iterable_response = self._generate_sync_stream(
                prompt=final_prompt,
                model=model # Uses instance default if None
            )

            async for text_chunk in self._consume_stream_async(sync_iterable_response):
                yield text_chunk
                full_llm_response_parts.append(text_chunk)

            # 6. Save LLM's full response (background)
            if conversation_id and full_llm_response_parts:
                full_response_text = "".join(full_llm_response_parts)
                if full_response_text.strip(): # Only save if there's content
                    background_tasks.add_task(
                        conversation_service.save_message,
                        conversation_id,
                        user_id, # Or a system/bot ID
                        "assistant",
                        full_response_text
                    )
            
            # 7. Perform summarization if needed (background)
            if conversation_id:
                # Pass self (LlmService instance) to the background task if it needs to call summarize_text_async
                background_tasks.add_task(
                    self._background_summarize_if_needed,
                    conversation_service,
                    conversation_id,
                    background_tasks # Pass background_tasks again if the helper itself adds more tasks (not typical)
                                     # Or just the services it needs.
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
        conversation_id: str, # Expecting string UUID from caller
    ):
        """Helper to run summarization in the background."""
        try:
            conv_id_uuid = uuid.UUID(conversation_id) # Convert string to UUID for service calls
            
            # should_summarize now returns: Tuple[bool, Optional[List[MessageModel]]]
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

    async def stream_llm_responses(
        self,
        user_query: str,
        user_id: str,
        conversation_service: "ConversationService", # Added
        background_tasks: "BackgroundTasks",       # Added
        conversation_id: Optional[str] = None,     # Added, assuming str or uuid.UUID
        character: Optional[str] = None,
        use_rag: bool = False,
        rag_k: int = 5,
        model: Optional[str] = None,
    ):
        """
        Async generator yields text chunks for a query. Handles conversation context,
        RAG, message saving, and background summarization.

        Args:
            user_query: The user's input query.
            user_id: The ID of the user.
            conversation_service: Initialized instance of ConversationService.
            background_tasks: FastAPI BackgroundTasks instance.
            conversation_id: Optional ID of the existing conversation.
            character: Optional character persona.
            use_rag: Flag to enable RAG using uploaded documents.
            rag_k: Number of document chunks to retrieve for RAG.
            model: LLM model name (defaults to instance default).
        Yields:
            str: Text chunks from the LLM response.
        """
        full_llm_response_parts = []
        try:
            # 1. Get or create conversation (assuming ConversationService handles this)
            #    And save user message in the background
            #    This part assumes ConversationService.handle_new_message or similar
            #    can create/get conversation and save the user message.
            #    For simplicity, let's assume conversation_id is managed and user message is saved
            #    before calling this, or ConversationService has a method to do it.
            #    Let's refine this: ConversationService should provide the conversation object.

            # For now, let's assume conversation_id is either provided or a new one is implicitly handled
            # by conversation_service when messages are saved or context is fetched.
            # A more robust approach would be:
            # conv_obj = await conversation_service.get_or_create_conversation(user_id, conversation_id, character)
            # current_conversation_id = conv_obj.id
            # background_tasks.add_task(conversation_service.save_message, current_conversation_id, user_id, "user", user_query)
            
            # Simplified: Assuming conversation_id is managed by the caller or a previous step.
            # If not, it needs to be created/retrieved here.
            # For this example, we'll assume `conversation_id` is valid if provided,
            # and `ConversationService` methods can handle `None` to create a new one if necessary.

            # Save user query (background)
            # This requires a valid conversation_id. If it's a new chat, it needs to be created first.
            # Let's assume `conversation_service.handle_user_message` does this and returns conv_id
            # For now, we'll proceed assuming conversation_id is available or handled by `save_message`
            if conversation_id: # Or handle creation if None
                 background_tasks.add_task(conversation_service.save_message, conversation_id, user_id, "user", user_query)


            # 2. Get previous conversation context
            conversation_history_context: Optional[List[Dict]] = None
            if conversation_id:
                try:
                    # Assuming get_context_for_llm is async
                    conversation_history_context = await conversation_service.get_context_for_llm(
                        conversation_id=conversation_id,
                        current_query_content=user_query # To potentially exclude it if already saved
                    )
                except Exception as e:
                    logger.error(f"Error retrieving conversation context for {conversation_id}: {e}", exc_info=True)
                    # Decide if to yield error or proceed without history

            # 3. Retrieve RAG context if enabled
            rag_context_docs: Optional[List[Dict]] = None
            if use_rag:
                try:
                    logger.info(f"Retrieving document chunks for RAG (user: {user_id}, k={rag_k})")
                    rag_context_docs = await self.vector_store_svc.search_relevant_messages(
                        conversation_id=conversation_id or '', # Use actual conv_id or default
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

            # 4. Prepare the final prompt
            final_prompt = self._prepare_prompt_and_context(
                user_query=user_query,
                conversation_history=conversation_history_context,
                rag_context_docs=rag_context_docs,
                character=character
            )

            # 5. Stream LLM response
            sync_iterable_response = self._generate_sync_stream(
                prompt=final_prompt,
                model=model # Uses instance default if None
            )

            async for text_chunk in self._consume_stream_async(sync_iterable_response):
                yield text_chunk
                full_llm_response_parts.append(text_chunk)

            # 6. Save LLM's full response (background)
            if conversation_id and full_llm_response_parts:
                full_response_text = "".join(full_llm_response_parts)
                if full_response_text.strip(): # Only save if there's content
                    background_tasks.add_task(
                        conversation_service.save_message,
                        conversation_id,
                        user_id, # Or a system/bot ID
                        "assistant",
                        full_response_text
                    )
            
            # 7. Perform summarization if needed (background)
            if conversation_id:
                # Pass self (LlmService instance) to the background task if it needs to call summarize_text_async
                background_tasks.add_task(
                    self._background_summarize_if_needed,
                    conversation_service,
                    conversation_id,
                    background_tasks # Pass background_tasks again if the helper itself adds more tasks (not typical)
                                     # Or just the services it needs.
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