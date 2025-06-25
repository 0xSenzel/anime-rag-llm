import logging
import requests
from typing import Optional, Tuple
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from google.oauth2 import id_token
from sqlalchemy.orm import Session
from app.models.profile import Profile
from app.schemas.auth import Token, TokenData, UserCreate
from app.core.config import settings
from app.core.security import create_access_token, verify_token
from google.auth.transport.requests import Request as GoogleRequest

logger = logging.getLogger(__name__)

class AuthService:
    def __init__(self, db: Session):
        self.db = db

    async def authenticate_google(self, code: str, redirect_uri: str) -> Tuple[Token, Profile]:
        """
        Authenticate a user with Google OAuth2.
        
        Args:
            code: The authorization code from Google
            redirect_uri: The redirect URI used in the OAuth flow
            
        Returns:
            Tuple[Token, User]: The access token and user object
            
        Raises:
            HTTPException: If authentication fails
        """
        try:
            # Exchange code for tokens
            token_response = requests.post(
                'https://oauth2.googleapis.com/token',
                data={
                    'code': code,
                    'client_id': settings.GOOGLE_CLIENT_ID,
                    'client_secret': settings.GOOGLE_CLIENT_SECRET,
                    'redirect_uri': redirect_uri,
                    'grant_type': 'authorization_code'
                }
            )
            token_data = token_response.json()
            
            if 'error' in token_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Google OAuth error: {token_data['error']}"
                )

            # Get user info from ID token
            id_info = id_token.verify_oauth2_token(
                token_data['id_token'],
                GoogleRequest(),
                settings.GOOGLE_CLIENT_ID
            )

            # Get or update user profile
            user = await self._get_or_update_profile(id_info)
            
            # Create access token
            access_token = create_access_token(
                data={"sub": user.id, "email": user.email}
            )

            return access_token, user

        except Exception as e:
            logger.error(f"Google authentication failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate Google credentials"
            )

    async def _get_or_update_profile(self, id_info: dict) -> Profile:
        """Get existing user profile or create new one from Google ID info."""
        user_id = id_info['sub']  # Google's user id, matches auth.users.id
        email = id_info['email']
        profile = self.db.query(Profile).filter(Profile.id == user_id).first()
        if not profile:
            # The trigger should have created the row, but if not, create it
            profile = Profile(id=user_id, email=email)
            self.db.add(profile)
            self.db.commit()
            self.db.refresh(profile)
        # Update profile fields if needed
        profile.full_name = id_info.get('name')
        profile.avatar_url = id_info.get('picture')
        self.db.commit()
        self.db.refresh(profile)
        return profile

    async def get_current_user(self, token: str) -> Profile:
        """
        Get the current user from the access token.
        
        Args:
            token: The access token
            
        Returns:
            User: The current user
            
        Raises:
            HTTPException: If token is invalid or user not found
        """
        try:
            payload = verify_token(token)
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials"
                )
            
            user = self.db.query(Profile).filter(Profile.id == user_id).first()
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            return user
            
        except Exception as e:
            logger.error(f"Token validation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
