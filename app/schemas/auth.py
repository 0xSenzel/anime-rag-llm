from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class TokenData(BaseModel):
    user_id: str
    email: str

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[HttpUrl] = None

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class GoogleAuthRequest(BaseModel):
    code: str
    redirect_uri: str
