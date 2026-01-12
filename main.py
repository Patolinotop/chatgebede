# ================================
# MENU EB – Backend Chatbot (Railway READY)
# Status: COMPLETO, CORRIGIDO e FUNCIONAL
# Autor: ChatGPT Cheats
# ================================

# ----------------
# requirements.txt
# ----------------
# flask
# openai
# python-dotenv
# requests

from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
import openai

# ================================
# ENV
# ================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # ex: Patolinotop/chatgebede
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada")
if not GITHUB_REPO:
    raise RuntimeError("GITHUB_REPO não configurado")

openai.api_key = OPENAI_API_KEY

# ================================
# APP
# ================================
app = Flask(__name__)

# ================================
# GitHub API (sem git)
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
                textos.append(r.text)
        except Exception:
            pass
    return "\n".join(textos)

# ================================
# OpenAI
# ================================

def gerar_resposta(pergunta: str, contexto: str) -> str:
    prompt = (
        "Use o contexto abaixo SOMENTE se for útil. "
        "Gere uma resposta curta, clara e gramatical. "
        "Máximo 100 caracteres.\n\n"
        f"CONTEXTO:\n{contexto}\n\n"
        f"PERGUNTA: {pergunta}\nRESPOSTA:" 
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()[:100]
    except Exception as e:
        return "Erro ao gerar resposta"

# ================================
# API
# ================================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    pergunta = data.get("input", "").strip()

    if not pergunta:
        return jsonify({"reply": "Entrada vazia"}), 400

    contexto = ler_contexto()
    resposta = gerar_resposta(pergunta, contexto)

    return jsonify({"reply": resposta})

# ================================
# START (Railway compatible)
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# ================================
# .env EXEMPLO
# ================================
# OPENAI_API_KEY=sk-xxxxxxxx
# GITHUB_REPO=Patolinotop/chatgebede
# GITHUB_BRANCH=main
