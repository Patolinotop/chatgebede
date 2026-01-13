# ================================
# MENU EB – Backend Chatbot API (FINAL HUMANIZADO)
# Objetivo:
# - Gerar texto curto (60–100 caracteres)
# - Humanizado, formal e gramatical
# - Usar .txt apenas como base semântica
# - Nunca devolver lixo de encoding ou texto gigante
# ================================

from flask import Flask, request, jsonify
import os, json, re, requests
from dotenv import load_dotenv
import openai

# ================================
# ENV
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # ex: Patolinotop/chatgebede
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes")

openai.api_key = OPENAI_API_KEY

# ================================
# APP
# ================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ================================
# Utils
# ================================

def limpar_texto(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    txt = re.sub(r"\n{2,}", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

# ================================
# GitHub – leitura dos .txt
# ================================

def listar_txt(path=""):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return []
    arquivos = []
    for item in r.json():
        if item.get("type") == "file" and item.get("name", "").endswith(".txt"):
            arquivos.append(item.get("download_url"))
        elif item.get("type") == "dir":
            arquivos.extend(listar_txt(item.get("path")))
    return arquivos


def ler_contexto():
    textos = []
    for url in listar_txt():
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                textos.append(limpar_texto(r.text))
        except Exception:
            pass
    return " ".join(textos)[:2500]

# ================================
# Geração de texto
# ================================

def gerar_resposta(tema: str, contexto: str) -> str:
    system_prompt = (
        "Você é um redator humano profissional. "
        "Escreva textos naturais, formais e bem pontuados. "
        "Não use gírias nem linguagem de IA."
    )

    user_prompt = f"""
TEMA: {tema}

BASE SEMÂNTICA:
{contexto}

INSTRUÇÕES:
- Um único parágrafo
- Entre 60 e 100 caracteres
- Frase completa
- Não citar documentos ou arquivos
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=90,
            temperature=0.35,
            presence_penalty=0.6,
            frequency_penalty=0.6,
            timeout=20
        )
        texto = resp.choices[0].message.content.strip()
    except Exception:
        texto = f"O tema {tema} envolve aspectos relevantes que exigem clareza e organização."

    if texto and texto[-1] not in ".!?":
        texto = texto.rsplit(" ", 1)[0] + "."

    return texto[:120]

# ================================
# API
# ================================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = data.get("input", "").strip()

    if not tema:
        return jsonify({"reply": "Tema não informado."})

    contexto = ler_contexto()
    resposta = gerar_resposta(tema, contexto)

    return app.response_class(
        response=json.dumps({"reply": resposta}, ensure_ascii=False),
        mimetype="application/json"
    )

# ================================
# START (Railway)
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
