# ================================
# MENU EB – Backend Chatbot (VERSÃO OTIMIZADA / CONTEXTUAL)
# Objetivo:
# - NÃO responder como chatbot genérico
# - SEM cortar resposta no meio
# - SEMPRE basear nos .txt quando possível
# - Texto curto (~100 caracteres) mas COMPLETO
# ================================

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
                textos.append(r.text.strip())
        except Exception:
            pass

    contexto = "\n".join(textos)
    return contexto[:4000]  # limite de segurança

# ================================
# OpenAI – geração CONTROLADA
# ================================

def gerar_resposta(tema: str, contexto: str) -> str:
    system_prompt = (
        "Você NÃO é um chatbot genérico. "
        "Você gera textos CURTOS, COESOS e COMPLETOS, "
        "baseados prioritariamente no CONTEXTO fornecido. "
        "Nunca corte frases no meio. "
        "Se o contexto não ajudar, gere um texto neutro e objetivo."
    )

    user_prompt = f"""
TEMA: {tema}

CONTEXTO (use somente se relevante):
{contexto}

INSTRUÇÕES:
- Gere um único parágrafo
- Máximo ~100 caracteres
- Frase completa (com ponto final)
- Linguagem neutra e clara
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=80,
            temperature=0.4,
        )

        texto = resp.choices[0].message.content.strip()

        # Segurança extra: evitar corte seco
        if texto and texto[-1] not in ".!?":
            texto = texto.rsplit(" ", 1)[0] + "."

        return texto

    except Exception:
        return "Não foi possível gerar o texto no momento."

# ================================
# API
# ================================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = data.get("input", "").strip()

    if not tema:
        return jsonify({"reply": "Tema vazio."}), 400

    contexto = ler_contexto()
    resposta = gerar_resposta(tema, contexto)

    return jsonify({"reply": resposta})

# ================================
# START (Railway)
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
