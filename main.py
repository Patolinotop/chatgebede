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
            "Responda com uma única frase curta (até 120 caracteres) sobre o tema abaixo.\n"
            f"Tema: {tema}\n"
            "Evite parágrafos ou explicações longas."
        )

        r = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": "Responda apenas com uma frase curta e direta."},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=120
        )

        # Se r.output for uma lista, pega o primeiro item
        if isinstance(r.output, list) and len(r.output) > 0:
            output = str(r.output[0]).strip()
        else:
            output = str(r.output).strip()

        return output[:140] if output else "Erro: resposta vazia."

    except Exception as e:
        return f"Erro: {str(e)}"
