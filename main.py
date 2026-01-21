from flask import Flask, request
import os, json, re, requests, traceback
import numpy as np
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from io import BytesIO

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

OPENAI_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes")

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

def limpar_texto(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def github_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def listar_arquivos(path=""):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, headers=github_headers(), timeout=20)
    if r.status_code != 200:
        return []

    data = r.json()
    arquivos = []

    for item in data:
        if item["type"] == "file" and item["name"].lower().endswith((".txt", ".pdf")):
            arquivos.append(item["download_url"])
        elif item["type"] == "dir":
            arquivos.extend(listar_arquivos(item["path"]))

    return arquivos

def ler_arquivo(url: str) -> str:
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return ""

    if url.lower().endswith(".txt"):
        return limpar_texto(r.text)

    if url.lower().endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(r.content))
            return limpar_texto(" ".join(p.extract_text() or "" for p in reader.pages))
        except Exception:
            return ""

    return ""

def chunkar(texto: str) -> List[str]:
    chunks = []
    i = 0
    while i < len(texto):
        chunks.append(texto[i:i+CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

_chunks = []
_vectors = None
_base_carregada = False

def carregar_base():
    global _chunks, _vectors, _base_carregada
    if _base_carregada:
        return

    arquivos = listar_arquivos()
    textos = []

    for url in arquivos:
        texto = ler_arquivo(url)
        for c in chunkar(texto):
            textos.append(f"FONTE: {url}\n{c}")

    if not textos:
        return

    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=textos
    )

    _chunks = textos
    _vectors = np.array([d.embedding for d in resp.data])
    _base_carregada = True

def buscar_contexto(pergunta: str) -> List[str]:
    carregar_base()
    if not _chunks:
        return []

    q_vec = client.embeddings.create(
        model=EMBED_MODEL,
        input=[pergunta]
    ).data[0].embedding

    q_vec = np.array(q_vec)

    sims = _vectors @ q_vec / (
        np.linalg.norm(_vectors, axis=1) * np.linalg.norm(q_vec)
    )

    idx = sims.argsort()[-TOP_K:][::-1]
    return [_chunks[i] for i in idx]

def gerar_resposta(pergunta: str, contexto: List[str]) -> str:
    base = "\n\n---\n\n".join(contexto)

    system = (
        "Use apenas o contexto fornecido. "
        "Se não houver informação suficiente, comece com 'Em geral,'."
    )

    user = f"""
PERGUNTA:
{pergunta}

CONTEXTO:
{base}

Responda em 1 parágrafo curto, em português, finalizando com ponto.
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_output_tokens=220
    )

    return limpar_texto(resp.output_text)

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json() or {}
        pergunta = data.get("input", "").strip()

        if not pergunta:
            return {"error": "tema_vazio"}

        contexto = buscar_contexto(pergunta)
        resposta = gerar_resposta(pergunta, contexto)

        return {"reply": resposta}

    except Exception as e:
        traceback.print_exc()
        return {"error": "server_failed", "detail": str(e)}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
