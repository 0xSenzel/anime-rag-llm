# app/services/conversation.py

import logging
import uuid
import asyncio # Added for async operations
from typing import List, Optional, Tuple, Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, select, and_, exc as sa_exc # Added exc
from datetime import datetime, timezone
from fastapi import BackgroundTasks

# Import models
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.summary import Summary
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LlmService

# --- Configuration ---
# Ideally, use Pydantic Settings (from pydantic_settings import BaseSettings)
# class ChatSettings(BaseSettings):
#     default_sliding_window_turns: int = 10
#     default_summary_turns_threshold: int = 20
#     default_summary_tokens_threshold: int = 2000 # Optional
#     max_context_tokens: int = 3500 # Example token limit for context assembly
#     vector_search_k: int = 5
#     # tokenizer_name: str = "cl100k_base"
# settings = ChatSettings()

# For simplicity here, using constants:
DEFAULT_SLIDING_WINDOW_TURNS = 10
DEFAULT_SUMMARY_TURNS_THRESHOLD = 20
DEFAULT_SUMMARY_TOKENS_THRESHOLD = 2000 # Optional
MAX_CONTEXT_TOKENS = 3500 # Example limit before LLM call
VECTOR_SEARCH_K = 5
# --- End Configuration ---

logger = logging.getLogger(__name__)

class ConversationService:
    """
    Handles logic related to conversations, messages, summaries,
    and context management for the LLM. Includes async operations.
    """
    def __init__(self, db: Session, vector_store: VectorStoreService, llm_service: LlmService):
        """
        Initializes the ConversationService.

        Args:
            db: SQLAlchemy database session.
            llm_service: An initialized instance of the LlmService for generating text and summaries.
        """
        self.db = db
        self.vector_store = vector_store
        self.llm_service = llm_service

    # --- Core Methods ---

    async def handle_new_message(
        self,
        background_tasks: BackgroundTasks,
        user_id: str,
        content: str,
        role: str,
        conversation_id: Optional[uuid.UUID] = None,
        character: Optional[str] = None,
        # Add other parameters like use_rag if needed by llm_service.stream_llm_responses
    ) -> Tuple[Conversation, Message, List[Dict]]: # Return convo, user msg, context used
        """
        Handles processing a new user message, saving it, preparing context,
        and triggering background tasks for embedding and potential summarization.
        """
        try:
            conversation = self.get_or_create_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
                character=character
            )
            # 1. Save User Message
            user_message = self.save_message(
                conversation_id=conversation.id,
                user_id=user_id,
                role=role,
                content=content
            )
            # 2. Trigger background task for user message embedding
            background_tasks.add_task(self.vector_store.add_message_embedding, message=user_message, user_id=user_id)

            # 3. Assemble Context for LLM
            llm_context = await self.get_context_for_llm(conversation.id, user_message.content)

            # 4. Check for Summarization (in background)
            # We pass the *conversation id* and let the background task re-check
            background_tasks.add_task(self._check_and_perform_summarization_async, conversation.id)

            return conversation, user_message, llm_context

        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error handling new message for user {user_id}: {e}", exc_info=True)
            self.db.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error handling new message for user {user_id}: {e}", exc_info=True)
            raise

    async def save_assistant_response(
        self,
        background_tasks: BackgroundTasks,
        conversation_id: uuid.UUID,
        user_id: str, # For verification/logging
        content: str,
    ) -> Message:
        """Saves the assistant's response and triggers embedding."""
        try:
            assistant_message = self.save_message(
                conversation_id=conversation_id,
                user_id=user_id, # Pass user_id for save_message logic/checks
                role="assistant",
                content=content
            )
            # Trigger background task for assistant message embedding
            background_tasks.add_task(self.vector_store.add_message_embedding, assistant_message, user_id=user_id)
            return assistant_message
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error saving assistant response for convo {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving assistant response for convo {conversation_id}: {e}", exc_info=True)
            raise

    # --- Context Assembly ---

    async def get_context_for_llm(self, conversation_id: uuid.UUID, current_query: str) -> List[Dict]:
        """
        Assembles the hybrid context (summary, recent messages, relevant history)
        for the LLM prompt, managing token limits.

        Returns:
            A list of dictionaries representing the context messages/summary,
            ordered chronologically. e.g., [{'role': 'summary', 'content': '...'},
            {'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        context_parts = []
        available_tokens = MAX_CONTEXT_TOKENS
        summary_tokens = 0

        # 1. Get Latest Summary
        latest_summary = self.get_latest_summary(conversation_id)
        summary_content = ""
        if latest_summary and latest_summary.summary_text:
            summary_content = f"Summary of earlier conversation:\n{latest_summary.summary_text}"
            summary_tokens = self.llm_service.count_tokens(summary_content)
            if summary_tokens <= available_tokens:
                context_parts.append({'role': 'system', 'content': summary_content}) # Use system role for summary
                available_tokens -= summary_tokens
            else:
                logger.warning(f"Summary for convo {conversation_id} too long ({summary_tokens} words > available {available_tokens}), skipping.")
                summary_tokens = 0 # Reset if skipped

        # 2. Get Recent Messages (Sliding Window)
        recent_messages = self.get_recent_messages(conversation_id, DEFAULT_SLIDING_WINDOW_TURNS)
        recent_messages_ids = {msg.id for msg in recent_messages}

        # 3. Vector Search for Relevant History (excluding recent messages)
        relevant_history_results = []
        if available_tokens > 100: # Only search if we have some token budget left
            try:
                relevant_history_results = await self.vector_store.search_relevant_messages(
                    conversation_id=conversation_id,
                    query_text=current_query,
                    k=VECTOR_SEARCH_K
                )
                # Filter out any results already in the sliding window
                relevant_history_results = [
                    res for res in relevant_history_results if res.get('id') not in recent_messages_ids # Safer get
                ]
                # Sort relevant results chronologically for inclusion
                relevant_history_results.sort(key=lambda x: x['created_at'])
            except Exception as e:
                logger.error(f"Vector search failed for convo {conversation_id}: {e}", exc_info=True)
                # Continue without vector context

        # 4. Combine Relevant History and Recent Messages, managing tokens
        # Prioritize adding relevant history first, then recent messages, trimming if needed
        combined_messages = []
        temp_token_count = 0

        # Add relevant history (older first)
        for res in relevant_history_results:
            msg_tokens = self.llm_service.count_tokens(res['content'])
            if temp_token_count + msg_tokens <= available_tokens:
                # Include essential fields for sorting and potential use
                combined_messages.append({
                    'role': res.get('role', 'unknown'), # Safer get
                    'content': res.get('content', ''),  # Safer get
                    'created_at': res.get('created_at', datetime.min.replace(tzinfo=timezone.utc)), # Provide default for sort
                    'id': res.get('id') # Keep ID for duplicate check if needed later
                })
                temp_token_count += msg_tokens
            else:
                logger.debug(f"Token limit reached while adding relevant history for convo {conversation_id}")
                break # Stop adding if token limit exceeded

        # Add recent messages (older first within the window)
        for msg in recent_messages:
            msg_tokens = self.llm_service.count_tokens(msg.content)
            # Check if this specific message ID is already included from relevant search (unlikely but possible)
            if msg.id not in {m.get('id') for m in combined_messages if m.get('id')}:
                 if temp_token_count + msg_tokens <= available_tokens:
                     combined_messages.append({'role': msg.role, 'content': msg.content, 'created_at': msg.created_at, 'id': msg.id})
                     temp_token_count += msg_tokens
                 else:
                     # If adding recent messages exceeds budget, we might stop or remove older ones from 'combined_messages'
                     # For simplicity, let's just stop adding more recent messages here.
                     logger.debug(f"Token limit reached while adding recent messages for convo {conversation_id}")
                     break

        # Ensure final combined messages are sorted correctly by timestamp
        combined_messages.sort(key=lambda x: x['created_at'])

        # Add the combined messages to the context parts
        # Remove internal keys like 'created_at', 'id' before final output
        context_parts.extend([{'role': m['role'], 'content': m['content']} for m in combined_messages])

        logger.info(f"Assembled context for convo {conversation_id}: Summary={bool(latest_summary)}, "
                    f"Relevant History Results={len(relevant_history_results)}, Recent Msgs Included={len(recent_messages)}, "
                    f"Final Context Parts={len(context_parts)}, Approx Tokens Used (Summary + Combined Msgs)={summary_tokens+temp_token_count}")

        # The context_parts list now contains the ordered context for the LLM
        # The calling function will format this list + the user query into the final prompt
        return context_parts


    # --- Summarization Logic (including async wrapper) ---

    async def _check_and_perform_summarization_async(self, conversation_id: uuid.UUID):
        """
        Background task: Checks if summarization is needed and performs it.
        """
        logger.info(f"Background task: Checking summarization for convo {conversation_id}")
        # TODO: Improve background task handling:
        # 1. Implement a task queue (e.g., Celery, Redis Queue (RQ), or FastAPI-Background-Tasks with a persistent backend)
        #    to ensure reliable execution and retries for summarization tasks.
        # 2. Consider caching summarization results (e.g., using Redis or FastAPI-Cache) to avoid redundant summarizations
        #    if the same conversation segment is encountered again.  Use a cache key based on conversation_id and a hash
        #    of the message content to ensure uniqueness.
        # 3. Implement proper session management for background tasks. If using FastAPI BackgroundTasks directly,
        #    use SessionLocal() from database.py to create a new session for each task. If using a task queue,
        #    rely on the queue's session management (Celery/RQ).
        try:
            needs_summary, messages = self.should_summarize(
                conversation_id=conversation_id,
                summary_turns_threshold=DEFAULT_SUMMARY_TURNS_THRESHOLD,
                summary_tokens_threshold=DEFAULT_SUMMARY_TOKENS_THRESHOLD
            )

            if needs_summary and messages:
                logger.info(f"Background task: Performing summary for convo {conversation_id}")
                summary_text = await self.perform_summary(messages)
                if summary_text:
                    summary_obj = self.save_summary(conversation_id, summary_text)
                    if summary_obj:
                        logger.info(f"Background task: Successfully saved summary {summary_obj.id}")
                else:
                     logger.warning(f"Background task: Summarization returned empty/failed for convo {conversation_id}")
            else:
                logger.info(f"Background task: No summarization needed for convo {conversation_id}")
            self.db.commit()
        except Exception as e:
             logger.error(f"Background task error during summarization check/perform for {conversation_id}: {e}", exc_info=True)
             self.db.rollback()


    def should_summarize(
        self,
        conversation_id: uuid.UUID,
        summary_turns_threshold: int = DEFAULT_SUMMARY_TURNS_THRESHOLD,
        summary_tokens_threshold: int = DEFAULT_SUMMARY_TOKENS_THRESHOLD,
    ) -> Tuple[bool, Optional[List[Message]]]:
        """
        Checks if a conversation needs summarization based on turn count OR
        token count thresholds.
        """
        try:
            last_summary_ts = self._get_latest_summary_timestamp(conversation_id)

            query_filter = [Message.conversation_id == conversation_id]
            if last_summary_ts:
                query_filter.append(Message.created_at > last_summary_ts)

            messages_since_last_summary = self.db.scalars(
                select(Message)
                .where(and_(*query_filter))
                .order_by(Message.created_at.asc()) # Chronological for token counting if needed
            ).all()

            if not messages_since_last_summary:
                return False, None # No messages since last summary

            num_messages = len(messages_since_last_summary)
            # A turn is often considered a user message + an assistant message
            # For simplicity, let's count raw messages for now, or adjust as num_messages // 2 for pairs.
            num_turns = num_messages // 2

            needs_summary_by_turns = num_turns >= summary_turns_threshold

            needs_summary_by_tokens = False
            total_tokens = 0
            if not needs_summary_by_turns and self.llm_service.tokenizer:
                try:
                    for msg in messages_since_last_summary:
                        # Sum tokens from message content
                        total_tokens += self.llm_service.count_tokens(msg.content)
                    logger.debug(f"Total tokens in messages since last summary: {total_tokens}")
                except Exception as e:
                    logger.error(f"Failed to tokenize content for convo {conversation_id}: {e}. Skipping token check.", exc_info=True)
                    needs_summary_by_tokens = False

            # 3. Combine conditions: Summarize if EITHER threshold is met
            needs_summary = needs_summary_by_turns or needs_summary_by_tokens

            if needs_summary:
                reason = f"Turns ({num_turns} >= {summary_turns_threshold})" if needs_summary_by_turns else f"Tokens ({total_tokens} >= {summary_tokens_threshold})"
                logger.info(f"Convo {conversation_id} needs summary. Reason: {reason}")
                return True, messages_since_last_summary
            else:
                return False, None
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error checking summarization for convo {conversation_id}: {e}", exc_info=True)
            return False, None


    async def perform_summary(self, messages_to_summarize: List[Message]) -> Optional[str]:
        """Generates a summary for the given messages using the LLM service."""
        if not messages_to_summarize:
            return None

        conversation_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in messages_to_summarize]
        )
        logger.info(f"Requesting summary for {len(messages_to_summarize)} messages.")

        try:
            full_summary = await self.llm_service.summarize_text_async(conversation_text)

            if full_summary:
                logger.info("Summary generated successfully.")
                return full_summary.strip()
            else:
                logger.warning("LLM returned an empty summary.")
                return None
        except Exception as e:
            logger.error(f"Error during LLM summarization call: {e}", exc_info=True)
            return None

    # --- Database Interaction Helpers (with basic error handling) ---

    def get_latest_summary_timestamp(self, conversation_id: uuid.UUID) -> Optional[datetime]:
        """Helper to get the creation timestamp of the most recent summary."""
        try:
            latest_summary_ts = self.db.scalar(
                select(func.max(Summary.created_at))
                .where(Summary.conversation_id == conversation_id)
            )
            return latest_summary_ts
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error getting latest summary timestamp for convo {conversation_id}: {e}", exc_info=True)
            return None

    def get_recent_messages(self, conversation_id: uuid.UUID, num_turns: int = DEFAULT_SLIDING_WINDOW_TURNS) -> List[Message]:
        """Retrieves the last N turns (2*N messages)."""
        num_messages = num_turns * 2
        if num_messages <= 0: return []
        try:
            recent_messages = self.db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(num_messages)
            ).all()
            return recent_messages[::-1] # Reverse for chronological order
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error getting recent messages for convo {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            return []

    def save_message(self, conversation_id: uuid.UUID, role: str, content: str, embedding_id: Optional[uuid.UUID] = None) -> Message:
        """Saves a message, updates conversation timestamp. Raises Exception on DB error."""
        try:
            conversation = self.db.get(Conversation, conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found for saving message.")
            # Optional: Add user check here if needed

            db_message = Message(conversation_id=conversation_id, role=role, content=content, embedding_id=embedding_id)
            self.db.add(db_message)

            conversation.updated_at = func.now()
            self.db.add(conversation)

            self.db.flush() # Flush to get ID and check constraints before returning
            logger.info(f"Saved message {db_message.id} for conversation {conversation_id}")
            return db_message
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error saving message for convo {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            raise

    def save_summary(self, conversation_id: uuid.UUID, summary_text: str) -> Optional[Summary]:
        """Saves a summary. Returns Summary object or None on failure."""
        try:
            # Ensure conversation exists and update its timestamp
            conversation = self.db.get(Conversation, conversation_id)
            if not conversation:
                logger.error(f"Cannot save summary, conversation {conversation_id} not found.")
                return None

            db_summary = Summary(conversation_id=conversation_id, summary_text=summary_text)
            self.db.add(db_summary)

            # Update conversation timestamp
            conversation.updated_at = func.now()
            self.db.add(conversation)

            self.db.flush() # Flush to get ID
            logger.info(f"Saved summary {db_summary.id} for conversation {conversation_id}")
            return db_summary
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error saving summary for convo {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            return None

    def get_latest_summary(self, conversation_id: uuid.UUID) -> Optional[Summary]:
        """Retrieves the most recent summary object."""
        try:
            summary = self.db.scalars(
                select(Summary)
                .where(Summary.conversation_id == conversation_id)
                .order_by(Summary.created_at.desc())
                .limit(1)
            ).first()
            return summary
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error getting latest summary for convo {conversation_id}: {e}", exc_info=True)
            self.db.rollback()
            return None

    def get_or_create_conversation(self, user_id: str, conversation_id: Optional[uuid.UUID] = None, character: Optional[str] = None) -> Conversation:
        """Gets or creates a conversation. Raises Exception on DB error."""
        try:
            if conversation_id:
                conversation = self.db.get(Conversation, conversation_id)
                if conversation:
                    # Optional: Verify user ownership
                    # if conversation.user_id != user_id: ... raise error ...
                    return conversation
                else:
                     logger.warning(f"Conversation ID {conversation_id} provided but not found. Creating new.")

            new_conversation = Conversation(user_id=user_id, character=character)
            self.db.add(new_conversation)

            self.db.flush() # Flush to get ID
            logger.info(f"Created new conversation {new_conversation.id} for user {user_id}")
            return new_conversation
        except sa_exc.SQLAlchemyError as e:
            logger.error(f"Database error getting/creating conversation for user {user_id}: {e}", exc_info=True)
            self.db.rollback()
            raise
