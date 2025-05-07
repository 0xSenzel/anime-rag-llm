# app/models/conversation.py
import uuid
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.database import Base
# Note: If you have a User model, you might import ForeignKey and relationship here
# from sqlalchemy import ForeignKey
# from sqlalchemy.orm import relationship

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
    # Note: server_onupdate requires specific handling in SQLAlchemy/Alembic or DB triggers.
    # Often, it's managed at the application level or via a simpler default.
    # For simplicity here, we'll use server_default like created_at,
    # but you might update it manually in your service logic upon adding messages/summaries.
    updated_at: DateTime = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Assuming user_id is a string UUID representation for flexibility,
    # adjust if you have a specific User model and want a real foreign key.
    user_id: str = Column(String, nullable=False, index=True)
    # If you have a User model:
    # user_id: uuid.UUID = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    # user = relationship("User") # Define relationship if needed

    character: str = Column(String(100), nullable=True)

    # Define relationships to messages and summaries (optional but useful)
    # messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    # summaries = relationship("Summary", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id='{self.user_id}', character='{self.character}')>"
