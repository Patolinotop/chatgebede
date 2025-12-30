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
            f"Gere uma única frase curta e direta (máx. 120 caracteres) sobre o tema abaixo.\n"
            f"Tema: {tema}\n"
            f"Arquivos no repositório: {repo_state}\n"
            f"Evite explicações. Apenas uma frase objetiva."
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Responda sempre com UMA frase curta e direta, sem explicação extra."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=50  # limite proposital pra evitar spam
        )

        reply = response.choices[0].message.content.strip()
        return reply[:140] if reply else "Erro: resposta vazia."

    except Exception as e:
        return f"Erro: {str(e)}"
