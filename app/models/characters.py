from sqlalchemy import Column, String, Text
from app.database import Base

class Character(Base):
    __tablename__ = "characters"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    bio = Column(Text, nullable=False)
    quotes = Column(Text, nullable=False)  # stored comma-separated
