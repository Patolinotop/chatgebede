# ================================
# MENU EB – Backend Chatbot (REPLIT READY)
# Modelo: gpt-4o-mini
# Objetivo: analisar TODOS os .txt do repositório e gerar texto natural
# Status: FINAL, AJUSTADO PARA USO POR TEMA
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
# GitHub API – ler TODOS os .txt
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
# OpenAI – geração por TEMA
# ================================

def gerar_resposta(tema: str, contexto: str) -> str:
    system_msg = (
        "Você escreve como um humano. "
        "NUNCA cite arquivos, fontes ou contexto. "
        "NÃO explique o processo. "
        "Use o contexto apenas como base de conhecimento silenciosa. "
        "Produza um texto curto, natural e fluido, como se alguém tivesse digitado. "
        "Máximo absoluto: 100 caracteres."
    )

    user_msg = (
        f"Tema: {tema}\n"
        "Escreva um texto curto e natural sobre o tema, usando o conhecimento implícito."
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "user", "content": f"CONHECIMENTO:\n{contexto}"}
            ],
            max_output_tokens=90,
            temperature=0.7,
        )

        text = resp.output_text
        if not text:
            return "Sem resposta"
        return text.strip()[:100]
    except Exception as e:
        print("[ERRO OpenAI]", e)
        return "Erro ao gerar resposta"

# ================================
# API
# ================================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = data.get("input", "").strip()

    if not tema:
        return jsonify({"reply": "Entrada vazia"}), 400

    contexto = ler_contexto()
    resposta = gerar_resposta(tema, contexto)

    return jsonify({"reply": resposta})

# ================================
# START (Replit)
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
