from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# 🔑 Cria o client usando a key do ambiente
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = FastAPI()

# 🌐 Libera acesso externo (Roblox / executores)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/chatgpt")
async def gerar_texto(tema: str = "tema aleatório"):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # modelo leve, rápido e barato
            messages=[
                {
                    "role": "user",
                    "content": f"Escreva um texto curto (máx. 2 linhas) sobre: {tema}"
                }
            ],
            max_tokens=60,
            temperature=0.7
        )

        texto = response.choices[0].message.content.strip()
        return texto

    except Exception as e:
        return f"Erro: {str(e)}"
