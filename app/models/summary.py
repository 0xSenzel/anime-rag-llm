import uuid
from sqlalchemy import Column, Integer, TEXT, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database import Base
# from sqlalchemy.orm import relationship # Uncomment if using relationships

class Summary(Base):
    """
    SQLAlchemy model for storing conversation summaries.
    """
    __tablename__ = "summaries"

    # Using Integer primary key for SERIAL equivalent
    id: int = Column(Integer, primary_key=True) # Autoincrements by default in PostgreSQL
    conversation_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    summary_text: str = Column(TEXT, nullable=False)

    # Timestamp
    created_at: DateTime = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Define relationship back to Conversation (optional but useful)
    # conversation = relationship("Conversation", back_populates="summaries")

    def __repr__(self):
        return f"<Summary(id={self.id}, convo_id={self.conversation_id})>"
