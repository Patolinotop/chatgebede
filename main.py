# ================================
# MENU EB – Backend Chatbot (VERSÃO FINAL ESTÁVEL)
# Correções:
# ✔ Erro intermitente "Não foi possível gerar o texto"
# ✔ Decodificação correta de unicode (acentos)
# ✔ Fallback quando OpenAI falhar
# ✔ Resposta SEMPRE baseada no contexto quando existir
# ✔ Texto curto, completo e nunca cortado
# ================================

from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
import openai
import json

# ================================
# ENV
# ================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes")

openai.api_key = OPENAI_API_KEY

# ================================
# APP
# ================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # <-- FIX unicode

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
    return contexto[:3500]

# ================================
# OpenAI – geração ROBUSTA
# ================================

def gerar_resposta(tema: str, contexto: str) -> str:
    system_prompt = (
        "Você é um gerador de textos curtos. "
        "Baseie-se PRIORITARIAMENTE no contexto fornecido. "
        "Nunca responda como chatbot genérico. "
        "Sempre produza uma frase completa e clara."
    )

    user_prompt = f"""
TEMA: {tema}

CONTEXTO:
{contexto}

INSTRUÇÕES:
- Um único parágrafo
- Máx. ~100 caracteres
- Frase completa com ponto final
- Português correto
"""

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=90,
            temperature=0.3,
            timeout=20
        )

        texto = resp.choices[0].message.content.strip()

    except Exception:
        # 🔁 FALLBACK: gerar texto simples a partir do contexto
        if contexto:
            frase = contexto.split(".")[0].strip()
            texto = frase + "." if frase else "Não foi possível gerar o texto no momento."
        else:
            texto = "Não foi possível gerar o texto no momento."

    # 🔒 Garantia de frase completa
    if texto and texto[-1] not in ".!?":
        texto = texto.rsplit(" ", 1)[0] + "."

    return texto

# ================================
# API
# ================================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = data.get("input", "").strip()

    if not tema:
        return jsonify({"reply": "Tema vazio."})

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
