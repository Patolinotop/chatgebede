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
            "Gere UMA única frase muito curta (máx. 120 caracteres) e direta sobre o tema abaixo.\n"
            "Evite parágrafos, não explique.\n"
            f"Tema: {tema}\n"
            f"Arquivos no repositório: {repo_state}"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=30  # ✅ removido temperature
        )

        texto = response.choices[0].message.content.strip()
        return texto[:140]

    except Exception as e:
        return f"Erro: {str(e)}"
