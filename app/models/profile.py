from sqlalchemy import Column, String, DateTime, Boolean, UUID, ForeignKey, Table
from sqlalchemy.sql import func
from app.database import Base
import uuid

# Register the external table
auth_users_table = Table(
    "users",
    Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    schema="auth",
    extend_existing=True
)

class Profile(Base):
    __tablename__ = "profiles"
    
    id = Column(
        UUID(as_uuid=True),
        ForeignKey("auth.users.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
        unique=True,
        index=True,
    )
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    avatar_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
