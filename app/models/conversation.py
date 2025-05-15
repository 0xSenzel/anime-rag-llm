# app/models/conversation.py
import uuid
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database import Base

class Conversation(Base):
    """
    SQLAlchemy model for storing conversation sessions.
    """
    __tablename__ = "conversations"

    id: uuid.UUID = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Timestamps
    created_at: DateTime = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: DateTime = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    user_id: str = Column(String, nullable=False, index=True)
    
    title: str = Column(String(200), nullable=True)
    character: str = Column(String(100), nullable=True)

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id='{self.user_id}', title='{self.title}', character='{self.character}')>"
