import logging
import os
import uuid
import asyncio
from typing import List, Optional, Any, Dict, Type
from datetime import datetime # Import datetime for parsing

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, PodSpec, Index # Import Index type
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
# Import necessary models for type hints and metadata structure
from app.models.message import Message

# --- Configuration Loading ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['VectorStoreService', 'create_default_embedding_model']

# --- Embedding Model Setup ---
# It's better practice to initialize the embedding model once and pass it
# to the service, rather than initializing it globally here.
# We'll define a default creation function for convenience if not passed.

def create_default_embedding_model():
    """Creates the default HuggingFace embedding model."""
    try:
        # Ensure VECTOR_DIMENSION matches this model
        logger.info("Initializing default embedding model: all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Adjust device ('cuda'/'mps' if available)
        )
    except Exception as e:
        logger.error(f"Failed to load default HuggingFace embedding model: {e}", exc_info=True)
        return None

DEFAULT_VECTOR_DIMENSION = 384 # Dimension for "all-MiniLM-L6-v2"

# --- VectorStoreService Class ---

class VectorStoreService:
    """
    Handles interactions with the vector database (Pinecone) for storing
    and retrieving message embeddings.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = DEFAULT_VECTOR_DIMENSION,
        metric: str = "cosine",
        use_serverless: bool = True,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        environment: Optional[str] = None, # Pod-based only
        pod_type: str = "p1.x1",         # Pod-based only
        embedding_model: Optional[Any] = None, # Allow injecting model
    ):
        """
        Initializes the VectorStoreService and connects to Pinecone.

        Args:
            api_key: Pinecone API Key.
            index_name: Name of the Pinecone index.
            dimension: Dimension of the vectors (must match embedding model).
            metric: Distance metric for Pinecone index.
            use_serverless: Whether to use a serverless Pinecone index.
            cloud: Cloud provider for serverless index (e.g., 'aws', 'gcp'). Required if use_serverless.
            region: Region for serverless index (e.g., 'us-east-1'). Required if use_serverless.
            environment: Environment for pod-based index. Required if not use_serverless.
            pod_type: Pod type for pod-based index.
            embedding_model: An initialized embedding model instance (e.g., HuggingFaceEmbeddings).
                               If None, attempts to create the default model.
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.use_serverless = use_serverless
        self.cloud = cloud
        self.region = region
        self.environment = environment
        self.pod_type = pod_type

        self.embedding_model = embedding_model or create_default_embedding_model()
        if not self.embedding_model:
             raise ValueError("VectorStoreService requires a valid embedding model.")
        if not self.dimension:
             # Attempt to get dimension if model provides it, otherwise raise error
             # This part depends on the embedding model interface. For HF, it's not direct.
             # Stick with requiring dimension passed correctly or using default.
             raise ValueError("VectorStoreService requires vector dimension.")


        self.client: Optional[Pinecone] = None
        self.index: Optional[Index] = None # Use specific Pinecone Index type hint
        self._connect() # Connect during initialization

    def _connect(self):
        """Establishes connection to Pinecone client and index."""
        if self.index:
            logger.debug("Pinecone connection already established.")
            return

        if not self.api_key:
            raise ValueError("Pinecone API key is required.")

        logger.info("Initializing Pinecone connection...")
        try:
            self.client = Pinecone(api_key=self.api_key)

            indexes = self.client.list_indexes()
            index_names = [idx["name"] for idx in indexes]

            if self.index_name not in index_names:
                self._create_index() # Create index if it doesn't exist
            else:
                logger.info(f"Index '{self.index_name}' already exists.")

            logger.info(f"Connecting to index '{self.index_name}'...")
            self.index = self.client.Index(self.index_name)
            # Optional: Describe index stats to confirm connection
            # logger.info(f"Index stats: {self.index.describe_index_stats()}")
            logger.info(f"Successfully connected to Pinecone index '{self.index_name}'.")

        except Exception as e:
            logger.error(f"Failed to initialize/connect to Pinecone index '{self.index_name}': {e}", exc_info=True)
            self.client = None
            self.index = None
            raise # Re-raise exception to signal connection failure

    def _create_index(self):
        """Creates the Pinecone index based on configuration."""
        if not self.client:
            raise ConnectionError("Pinecone client not initialized before creating index.")

        logger.info(f"Creating new Pinecone index '{self.index_name}'...")
        spec: Any
        if self.use_serverless:
            if not self.cloud or not self.region:
                raise ValueError("Cloud and Region are required for serverless index.")
            spec = ServerlessSpec(cloud=self.cloud, region=self.region)
            logger.info(f"Using Serverless spec: {spec}")
        else:
            if not self.environment:
                raise ValueError("Environment is required for pod-based index.")
            spec = PodSpec(environment=self.environment, pod_type=self.pod_type)
            logger.info(f"Using Pod spec: {spec}")

        self.client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=spec
        )
        logger.info(f"Index '{self.index_name}' creation initiated. May take time to initialize fully.")
        # Consider adding a wait loop here in production:
        # while not self.client.describe_index(self.index_name).status['ready']:
        #     logger.info("Waiting for index to become ready...")
        #     asyncio.sleep(5) # Use asyncio.sleep if in async context

    async def _execute_blocking(self, func, *args, **kwargs):
        """Helper to run blocking functions in a separate thread."""
        # Use asyncio.to_thread for cleaner execution of blocking code
        return await asyncio.to_thread(func, *args, **kwargs)

    async def add_message_embedding(self, message: Message, user_id: str):
        """
        Generates embedding for a message and upserts it into Pinecone.

    Args:
            message: The SQLAlchemy Message object.
            user_id: The ID of the user associated with the conversation.
                     Passed explicitly to avoid DB lookups here.
        """
        if not self.embedding_model:
            logger.error("Embedding model not available.")
            return
        if not self.index:
            logger.error(f"Pinecone index not available. Cannot add embedding for message {message.id}.")
            # Optionally attempt self._connect() here if lazy connection is desired
            return

        logger.debug(f"Preparing to add embedding for message {message.id}")

        try:
            # 1. Generate Embedding
            content_to_embed = [message.content]
            try:
                embedding_result = await self._execute_blocking(
                    self.embedding_model.embed_documents, content_to_embed
                )
                if not embedding_result or not embedding_result[0]:
                    logger.error(f"Failed to generate embedding for message {message.id}")
                    return
                vector = embedding_result[0]
            except Exception as embed_e:
                 logger.error(f"Error generating embedding for message {message.id}: {embed_e}", exc_info=True)
                 return

            # 2. Prepare Metadata
            metadata = {
                "conversation_id": str(message.conversation_id),
                "user_id": str(user_id), # Use explicitly passed user_id
                "role": message.role,
                "created_at": message.created_at.isoformat(),
                "content": message.content # Store original content
            }

            # 3. Prepare Vector for Upsert
            vector_id = str(message.id) # Use message UUID as the Pinecone vector ID
            vector_to_upsert = (vector_id, vector, metadata)

            # 4. Upsert to Pinecone
            logger.debug(f"Upserting vector for message {vector_id}...")
            try:
                upsert_response = await self._execute_blocking(
                   self.index.upsert, vectors=[vector_to_upsert]
                )
                logger.info(f"Pinecone upsert response for message {vector_id}: {upsert_response}")
            except Exception as upsert_e:
                 logger.error(f"Error upserting vector for message {vector_id}: {upsert_e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error in add_message_embedding for message {message.id}: {e}", exc_info=True)

    async def add_document_chunks(
        self,
        documents: List[Document],
        user_id: str,
        conversation_id: uuid.UUID,
        batch_size: int = 100
    ) -> bool:
        """
        Generates embeddings for document chunks and upserts them into Pinecone in batches.

        Args:
            documents: List of LangChain Document objects.
            user_id: The ID of the user uploading the documents.
            batch_size: Number of vectors to upsert in each batch.

        Returns:
            True if all batches were processed successfully, False otherwise.
        """
        if not self.embedding_model:
            logger.error("Embedding model not available.")
            return False
        if not self.index:
            logger.error("Pinecone index not available. Cannot add document chunks.")
            return False
        if not documents:
            logger.info("No document chunks provided to add.")
            return True

        logger.info(f"Processing {len(documents)} document chunks for user {user_id} at conversation {conversation_id} in batches of {batch_size}.")
        all_successful = True
        processed_batches = 0
        total_vectors_processed = 0

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [
                {
                    "user_id": str(user_id),
                    "conversation_id": str(conversation_id),
                    "document_id": str(doc.metadata.get("source", f"doc_{i}")),
                    "chunk_index": i,
                    "text": doc.page_content[:500],  # Store partial text for context
                    **doc.metadata  # Optionally include all metadata
                }
                for i, doc in enumerate(documents)
            ]

            for i in range(0, len(documents), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                batch_indices = list(range(i, min(i + batch_size, len(documents))))

                if not batch_texts:
                    continue

                try:
                    embeddings = await self._execute_blocking(
                        self.embedding_model.embed_documents, batch_texts
                    )
                    if len(embeddings) != len(batch_texts):
                        logger.error(f"Embedding count mismatch in batch {processed_batches + 1}.")
                        all_successful = False
                        continue
                except Exception as embed_e:
                    logger.error(f"Error generating embeddings for batch {processed_batches + 1}: {embed_e}", exc_info=True)
                    all_successful = False
                    continue

                vectors_to_upsert = []
                for idx, (embedding, metadata) in enumerate(zip(embeddings, batch_metadatas)):
                    original_index = batch_indices[idx]
                    doc_identifier = str(metadata.get("document_id", "unknown"))
                    vector_id = f"doc::{user_id}::{doc_identifier}::chunk_{original_index}"
                    vectors_to_upsert.append((vector_id, embedding, metadata))

                try:
                    upsert_response = await self._execute_blocking(
                        self.index.upsert, vectors=vectors_to_upsert
                    )
                    logger.info(f"Pinecone upsert response for batch {processed_batches + 1}: {upsert_response}")
                    total_vectors_processed += len(vectors_to_upsert)
                except Exception as upsert_e:
                    logger.error(f"Error upserting batch {processed_batches + 1}: {upsert_e}", exc_info=True)
                    all_successful = False

                processed_batches += 1

            logger.info(f"Finished processing document chunks for user {user_id}. Total vectors processed: {total_vectors_processed}. Success: {all_successful}")
            return all_successful

        except Exception as e:
            logger.error(f"Unexpected error during add_document_chunks for user {user_id}: {e}", exc_info=True)
            return False

    async def search_relevant_messages(
        self,
        conversation_id: uuid.UUID,
        query_text: str,
        k: int = 5,
        min_score: float = 0.6,
    ) -> List[Dict]:
        """
        Searches for messages relevant to the query within a specific conversation.

        Args:
            conversation_id: The ID of the conversation to search within.
            query_text: The text to search for relevance.
            k: The maximum number of relevant messages to return.
            min_score: Minimum similarity score to consider a result relevant.

        Returns:
            A list of dictionaries, each representing a relevant message with
            its metadata (id, content, role, created_at) and similarity score.
            Returns an empty list on failure or if no relevant results are found.
        """
        results = []
        if not self.embedding_model:
            logger.error("Embedding model not available.")
            return results
        if not self.index:
            logger.error(f"Pinecone index not available. Cannot search messages for conversation {conversation_id}.")
            return results

        logger.debug(f"Searching for {k} relevant messages in conversation {conversation_id} with min_score={min_score}")

        try:
            # 1. Generate Query Embedding
            query_vector = await self._execute_blocking(
                self.embedding_model.embed_query, query_text
            )

            # 2. Define Pinecone Filter
            pinecone_filter = {"conversation_id": str(conversation_id)}
            logger.info(f"Querying Pinecone index '{self.index_name}' with filter: {pinecone_filter}")

            # 3. Query Pinecone
            query_response = await self._execute_blocking(
                self.index.query,
                vector=query_vector,
                filter=pinecone_filter,
                top_k=k,
                include_metadata=True
            )
            logger.debug(f"Pinecone query raw response: {query_response}")

            # 4. Process Results
            if query_response and hasattr(query_response, 'matches') and query_response.matches:
                logger.info(f"Found {len(query_response.matches)} matches in Pinecone response for convo {conversation_id}.")
                for match in query_response.matches:
                    score = getattr(match, 'score', 0.0)
                    if score < min_score:
                        logger.info(f"Skipping match {getattr(match, 'id', None)} with low score {score:.4f}")
                        continue  # Skip results below the threshold

                    metadata = getattr(match, 'metadata', None)
                    vector_id_str = getattr(match, 'id', None)

                    content = metadata.get('content') if metadata else None
                    role = metadata.get('role') if metadata else None
                    created_at_iso = metadata.get('created_at') if metadata else None

                    if not all([content, role, created_at_iso, vector_id_str]):
                        continue

                    try:
                        created_at_dt = datetime.fromisoformat(created_at_iso)
                        message_id = uuid.UUID(vector_id_str)
                    except Exception:
                        continue

                    results.append({
                        "id": message_id,
                        "content": content,
                        "role": role,
                        "created_at": created_at_dt,
                        "score": score,
                    })
                logger.info(f"Processed {len(results)} relevant messages for conversation {conversation_id} after score filtering.")
            else:
                logger.warning(f"No 'matches' found in Pinecone query response for conversation {conversation_id}.")

        except Exception as e:
            logger.error(f"Unexpected error during search_relevant_messages for conversation {conversation_id}: {e}", exc_info=True)
            return []

        return results

    def close(self):
        """Clean up resources (if Pinecone client requires it)."""
        # Check Pinecone client documentation for specific cleanup needs.
        # Newer versions might manage connections automatically.
        logger.info("VectorStoreService closing (Pinecone client cleanup if applicable).")
        self.index = None
        self.client = None # Allow garbage collection

