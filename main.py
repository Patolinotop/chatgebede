# 🔧 Backend Chatbot API – VERSÃO CORRIGIDA (SEM GIT)
# Resolve erro: "Bad git executable"
# Usa a API do GitHub para ler arquivos .txt (não precisa de git instalado)

# =========================
# requirements.txt
# =========================
# flask
# openai
# python-dotenv
# requests

from flask import Flask, request, jsonify
import os
import requests
import openai
from dotenv import load_dotenv

# 🔒 Variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")  # ex: Patolinotop/chatgebede
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# =========================
# GitHub helpers (sem git)
# =========================

def listar_arquivos_txt():
    """Lista recursivamente arquivos .txt via GitHub Contents API"""
    base_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents"
    arquivos = []

    def walk(path=""):
        url = f"{base_url}/{path}" if path else base_url
        r = requests.get(url)
        if r.status_code != 200:
            return
        for item in r.json():
            if item.get("type") == "file" and item.get("name", "").endswith(".txt"):
                arquivos.append(item.get("download_url"))
            elif item.get("type") == "dir":
                walk(item.get("path"))

    walk()
    return arquivos


def ler_contexto_txt():
    textos = []
    for url in listar_arquivos_txt():
        r = requests.get(url)
        if r.status_code == 200:
            textos.append(r.text)
    return "\n".join(textos)

# =========================
# OpenAI
# =========================

def gerar_resposta(pergunta, contexto):
    prompt = (
        "Use o contexto abaixo (se for útil) e gere uma resposta curta, correta e clara. "
        "Máximo de 100 caracteres.\n\n"
        f"CONTEXTO:\n{contexto}\n\n"
        f"PERGUNTA: {pergunta}\nRESPOSTA:" 
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()[:100]
    except Exception as e:
        return f"Erro ao gerar resposta"

# =========================
# API
# =========================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    pergunta = data.get("input", "").strip()
    if not pergunta:
        return jsonify({"reply": "Entrada vazia."}), 400

    contexto = ler_contexto_txt()
    resposta = gerar_resposta(pergunta, contexto)
    return jsonify({"reply": resposta})

# =========================
# Start
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# =========================
# .env EXEMPLO
# =========================
# OPENAI_API_KEY=sk-xxxx
# GITHUB_REPO=Patolinotop/chatgebede
# GITHUB_BRANCH=main
