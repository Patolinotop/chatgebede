import asyncio
from fastapi import FastAPI
from bot import start_discord_bot

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_discord_bot())
