# ================================
# MENU EB – Backend Chatbot API (OPENAI >=1.0 CORREÇÃO DEFINITIVA)
# FIX CRÍTICO:
# ✔ Corrige leitura do objeto Response (NÃO é dict)
# ✔ Remove uso incorreto de item.get
# ✔ Extrai texto pelo atributo oficial: response.output_text
# ✔ Mantém DEBUG TOTAL
# ================================

from flask import Flask, request
import os, json, re, requests, traceback
from dotenv import load_dotenv
from openai import OpenAI

# ================================
# ENV
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================================
# APP
# ================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ================================
# Utils
# ================================

def limpar_texto(txt: str) -> str:
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
        print("[DEBUG] GitHub API status", r.status_code)
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
            print("[DEBUG] TXT fetch", url, r.status_code)
            if r.status_code == 200:
                textos.append(limpar_texto(r.text))
        except Exception as e:
            print("[DEBUG] TXT erro", e)
    contexto = " ".join(textos)
    print("[DEBUG] Contexto size", len(contexto))
    return contexto[:2500]

# ================================
# OpenAI (API NOVA – CORRETA)
# ================================

def gerar_resposta(tema: str, contexto: str):
    system_prompt = (
        "Você é um redator humano profissional. "
        "Escreva um texto curto, formal e gramatical. "
        "Não use gírias."
    )

    user_prompt = (
        f"TEMA: {tema}\n"
        f"BASE SEMÂNTICA: {contexto}\n"
        "Gere um único parágrafo entre 60 e 100 caracteres, frase completa."
    )

    try:
        print("[DEBUG] Chamando OpenAI (Responses)")

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_output_tokens=120,
        )

        # ✅ FORMA CORRETA (documentada)
        texto = response.output_text

        if not texto:
            raise RuntimeError("Resposta vazia da OpenAI")

        print("[DEBUG] OpenAI OK")
        return texto.strip(), None

    except Exception as e:
        print("[DEBUG] OpenAI ERRO")
        traceback.print_exc()
        return None, str(e)

# ================================
# API
# ================================

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = data.get("input", "").strip()

    if not tema:
        return app.response_class(
            response=json.dumps({"error": "tema_vazio"}, ensure_ascii=False),
            mimetype="application/json"
        )

    print("[DEBUG] Tema:", tema)

    contexto = ler_contexto()
    texto, erro = gerar_resposta(tema, contexto)

    if erro:
        return app.response_class(
            response=json.dumps({
                "error": "openai_failed",
                "detail": erro,
                "context_size": len(contexto)
            }, ensure_ascii=False),
            mimetype="application/json"
        )

    if texto and texto[-1] not in ".!?":
        texto = texto.rsplit(" ", 1)[0] + "."

    return app.response_class(
        response=json.dumps({"reply": texto}, ensure_ascii=False),
        mimetype="application/json"
    )

# ================================
# START (Railway)
# ================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
