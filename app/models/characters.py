from sqlalchemy import Column, String, Text, JSON, DateTime
from sqlalchemy.sql import func
from app.database import Base
import uuid

class Character(Base):
    __tablename__ = "characters"
    
    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, index=True, nullable=False)
    aliases = Column(JSON, nullable=True)
    description = Column(Text, nullable=False)
    backstory = Column(Text, nullable=False)
    motivations = Column(Text, nullable=False)
    catchphrases = Column(JSON, nullable=True)
    sample_dialogues = Column(JSON, nullable=True)
    knowledge_domains = Column(JSON, nullable=True)
    avatar_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
