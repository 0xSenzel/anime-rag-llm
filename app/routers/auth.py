from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from app.core.dependencies import get_db
from app.services.auth import AuthService
from app.schemas.auth import Token, UserResponse, GoogleAuthRequest

router = APIRouter(prefix="/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

@router.post("/google", response_model=Token)
async def google_auth(
    request: GoogleAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate with Google OAuth2.
    
    Args:
        request: The Google auth request containing the code and redirect URI
        
    Returns:
        Token: The access token and related data
    """
    auth_service = AuthService(db)
    token, user = await auth_service.authenticate_google(
        code=request.code,
        redirect_uri=request.redirect_uri
    )
    return token

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get the current authenticated user.
    
    Args:
        token: The access token
        
    Returns:
        UserResponse: The current user's data
    """
    auth_service = AuthService(db)
    user = await auth_service.get_current_user(token)
    return user

# test
# https://accounts.google.com/o/oauth2/v2/auth?response_type=code
# &client_id=722344184986-2blm27kbf86s9ntgil7lk4c95nnefi4j.apps.googleusercontent.com
# &redirect_uri=http://localhost:3000/auth/callback
# &scope=openid%20email%20profile
# &access_type=offline
# &prompt=consent