from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-5-nano"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def scan_repository(base_path="."):
    files_info = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.startswith("."):
                continue
            path = os.path.join(root, file)
            try:
                size = os.path.getsize(path)
                files_info.append(f"{file} ({size} bytes)")
            except:
                pass
    return ", ".join(files_info)[:1000]

@app.get("/api/chatgpt")
async def gerar_texto(tema: str = "tema"):
    try:
        repo_state = scan_repository()

        prompt = (
            f"Gere uma única frase bem curta e direta sobre o tema abaixo.\n"
            f"Tema: {tema}\n"
            f"Arquivos no projeto: {repo_state[:500]}\n"
            f"Nada de parágrafo, apenas uma frase."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Responda sempre com apenas UMA frase curta e direta, no máximo 120 caracteres. Sem explicações."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        reply = response.choices[0].message.content or ""
        return reply[:140] if reply else "Erro: resposta vazia."

    except Exception as e:
        return f"Erro: {str(e)}"
