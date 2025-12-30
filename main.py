from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import os

# 🔑 OpenAI API Key do ambiente (definida no Railway)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# ✅ CORS liberado pro Roblox
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir requisições externas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/chatgpt")
async def gerar_texto(tema: str = "tema aleatório"):
    prompt = f"Escreva um pequeno texto (máx. 2 linhas) sobre: {tema}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.7
        )
        texto = response["choices"][0]["message"]["content"].strip()
        return texto
    except Exception as e:
        return f"Erro: {str(e)}"
