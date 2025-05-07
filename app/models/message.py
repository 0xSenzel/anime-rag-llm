# app/models/message.py
import uuid
from sqlalchemy import Column, Index, String, TEXT, DateTime, CheckConstraint, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database import Base
# from sqlalchemy.orm import relationship # Uncomment if using relationships

class Message(Base):
    """
    SQLAlchemy model for storing individual messages within a conversation.
    """
    __tablename__ = "messages"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    role: str = Column(String(10), nullable=False) # 'user', 'assistant', 'system'
    content: str = Column(TEXT, nullable=False)

    # Timestamp
    created_at: DateTime = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Optional link to vector store ID
    embedding_id: uuid.UUID = Column(UUID(as_uuid=True), nullable=True, index=True)

    # Define constraints and indexes
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="message_role_check"),
        # Composite index for efficient retrieval of messages per conversation ordered by time
        # Useful for sliding window implementation
        Index('idx_messages_conv_id_created_at_desc', conversation_id, created_at.desc()),
    )

    # Define relationship back to Conversation (optional but useful)
    # conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self):
        return f"<Message(id={self.id}, convo_id={self.conversation_id}, role='{self.role}')>"
