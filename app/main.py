from fastapi import FastAPI
from app.core.logging_config import setup_logging

setup_logging()
app = FastAPI()

from app.routers import characters as characters_router
from app.routers import upload, chat
from app.database import Base, engine
from app.routers import auth

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Anime Knowledge Base API")

app.include_router(characters_router.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(auth.router)

@app.get("/")
async def root():
    return {"message": "Hello, Anime Knowledge Base here!"}