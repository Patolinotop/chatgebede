# webservice.py
import os
import asyncio
from fastapi import FastAPI
from bot import start_discord_bot

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
async def startup_event():
    # roda o bot em background no mesmo processo do web service
    asyncio.create_task(start_discord_bot())
