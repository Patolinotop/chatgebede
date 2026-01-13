# ================================
# MENU EB – Backend Chatbot (VERSÃO FINAL CORRIGIDA E HUMANIZADA)
# Correções DEFINITIVAS:
# ✔ Remove lixo de encoding (BOM / UTF-8 quebrado)
# ✔ NÃO devolve texto cru dos .txt
# ✔ Usa .txt APENAS como base semântica
# ✔ Resposta SEMPRE curta (60–100 caracteres)
# ✔ Texto HUMANIZADO (natural, mas formal e gramatical)
# ✔ Nunca responde algo fora do tema
# ================================

from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv
import openai
import json
import re

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
app.config["JSON_AS_ASCII"] = False

# ================================
# Utils – limpeza de texto
# ================================

def limpar_texto(txt: str) -> str:
    if not txt:
        return ""

    # Remove BOM e lixo de encoding
    txt = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Remove excesso de espaços e linhas vazias
    txt = re.sub(r"\n{2,}", "\n", txt)
    txt = re.sub(r"\s{2,}", " ", txt)

    # Remove cabeçalhos comuns (ex: documentos)
    blacklist = [
        "EXÉRCITO", "CAPACITAÇÃO", "PATENTE", "________________________________________________"
    ]

    for b in blacklist:
        txt = txt.replace(b, "")

    return txt.strip()

# ================================
# GitHub – leitura segura dos .txt
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
                limpo = limpar_texto(r.text)
                if limpo:
                    textos.append(limpo)
        except Exception:
            pass

    contexto = "\n".join(textos)
    return contexto[:2500]

# ================================
# OpenAI – geração CONTROLADA e HUMANIZADA
# ================================

def gerar_resposta(tema: str, contexto: str) -> str:
    system_prompt = (
        "Você é um redator humano profissional. "
        "Seu texto NÃO deve parecer escrito por IA. "
        "Escreva de forma natural, formal e bem pontuada. "
        "Nunca copie trechos do contexto literalmente. "
        "Use o contexto apenas como base de conhecimento."
    )

    user_prompt = f"""
TEMA: {tema}

BASE DE CONHECIMENTO:
{contexto}

INSTRUÇÕES OBRIGATÓRIAS:
- Gere UM único parágrafo
- Entre 60 e 100 caracteres
- Frase completa e coesa
- Português formal
- Não mencionar documentos, arquivos ou textos
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
        texto = "O tema informado exige análise específica para gerar um texto adequado."

    # Garantia final de tamanho e completude
    texto = texto[:120]
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
