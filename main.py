# ================================
# MENU EB – Backend Chatbot (REPLIT READY)
# Modelo: gpt-4o-mini (boa qualidade + custo baixo)
# Status: COMPLETO, ATUALIZADO e FUNCIONAL
# ================================

# ----------------
# requirements.txt
# ----------------
# flask
# openai>=1.0.0
# python-dotenv
# requests

from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

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

client = OpenAI(api_key=OPENAI_API_KEY)

# ================================
# APP
# ================================
app = Flask(__name__)

# ================================
# GitHub API (ler .txt como contexto)
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
# OpenAI – geração (gpt-4o-mini)
# ================================

def gerar_resposta(pergunta: str, contexto: str) -> str:
    # Prompt curto e seguro para Roblox (<=100 chars)
    system_msg = (
        "Você é um assistente que escreve respostas curtas, claras e gramaticais. "
        "Use o contexto apenas se ajudar. Limite a resposta a no máximo 100 caracteres."
    )

    user_msg = f"CONTEXTO:\n{contexto}\n\nPERGUNTA: {pergunta}"

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_output_tokens=80,
            temperature=0.6,
        )

        # Extrai texto do response (compatível com SDK novo)
        text = resp.output_text
        if not text:
            return "Sem resposta"
        return text.strip()[:100]
    except Exception as e:
        # Log simples para debug no Replit
        print("[ERRO OpenAI]", e)
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
# START (Replit / local)
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)

# ================================
# .env (exemplo)
# ================================
# OPENAI_API_KEY=sk-xxxxxxxx
# GITHUB_REPO=Patolinotop/chatgebede
# GITHUB_BRANCH=main
