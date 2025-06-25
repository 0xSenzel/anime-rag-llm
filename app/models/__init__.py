# app/models/__init__.py

# Import the Base object - adjust the path if your Base is elsewhere
from app.database import Base

# Import all your model classes defined in this directory
from .characters import Character
from .conversation import Conversation
from .message import Message
from .summary import Summary
from .profile import Profile
# Add imports for any other models you create here in the future
