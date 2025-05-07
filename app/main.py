from fastapi import FastAPI
from app.routers import characters as characters_router
from app.routers import upload, chat
from app.database import Base, engine

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Anime Knowledge Base API")

app.include_router(characters_router.router)
app.include_router(upload.router)
app.include_router(chat.router)

@app.get("/")
async def root():
    return {"message": "Hello, Anime Knowledge Base here!"}