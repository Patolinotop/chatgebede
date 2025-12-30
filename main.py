from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# ===============================
# CONFIG OPENAI
# ===============================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = "gpt-5-nano"

# ===============================
# FASTAPI
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# FUNÇÃO: VARRER REPOSITÓRIO
# ===============================
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

# ===============================
# ROTA PRINCIPAL
# ===============================
@app.get("/api/chatgpt")
async def gerar_texto(tema: str = "tema"):
    try:
        repo_state = scan_repository()

        prompt = (
            "Responda com uma frase única, curta e direta (máx. 120 caracteres) sobre o tema:\n"
            f"{tema}\n"
            "Não use explicações longas, apenas a frase limpa e direta."
        )

        r = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": "Responda apenas texto limpo, curto, direto."},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=120
        )

        output = r.output.strip()
        return output[:140] if output else "Erro: resposta vazia."

    except Exception as e:
        return f"Erro: {str(e)}"
