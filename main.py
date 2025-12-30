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

MODEL_NAME = "gpt-5-nano"  # conforme solicitado

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
    return ", ".join(files_info)[:1000]  # limite de segurança


# ===============================
# ROTA PRINCIPAL
# ===============================
@app.get("/api/chatgpt")
async def gerar_texto(tema: str = "tema"):
    try:
        repo_state = scan_repository()

        prompt = (
            "Gere UMA frase extremamente curta (máx. 120 caracteres), clara e direta.\n"
            "NÃO use parágrafos, NÃO use explicações.\n"
            f"Tema: {tema}\n"
            f"Estado do repositório (verificação interna): {repo_state}"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=25,          # MUITO curto
            temperature=0.5
        )

        texto = response.choices[0].message.content.strip()

        # Garantia extra de tamanho (anti-Roblox)
        return texto[:140]

    except Exception as e:
        return f"Erro: {str(e)}"
