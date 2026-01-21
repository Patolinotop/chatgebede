from flask import Flask, request
import os, json, re, requests, traceback, time, hashlib
import numpy as np
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBED_MODEL = "text-embedding-3-large"

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 6

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

def limpar_texto(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def _github_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def listar_arquivos(path=""):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, headers=_github_headers())
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
    r = requests.get(url)
    if r.status_code != 200:
        return ""

    if url.lower().endswith(".txt"):
        return limpar_texto(r.text)

    if url.lower().endswith(".pdf"):
        reader = PdfReader(r.content)
        pages = [p.extract_text() or "" for p in reader.pages]
        return limpar_texto(" ".join(pages))

    return ""

def chunkar(texto: str) -> List[str]:
    chunks = []
    i = 0
    while i < len(texto):
        chunk = texto[i:i + CHUNK_SIZE]
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embedding(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return np.array([d.embedding for d in resp.data])

_cache_chunks = []
_cache_vectors = None

def carregar_base():
    global _cache_chunks, _cache_vectors
    if _cache_chunks:
        return

    arquivos = listar_arquivos()
    textos = []

    for url in arquivos:
        t = ler_arquivo(url)
        for c in chunkar(t):
            textos.append(f"FONTE: {url}\n{c}")

    _cache_chunks = textos
    _cache_vectors = embedding(textos)

def buscar_contexto(pergunta: str) -> List[str]:
    carregar_base()
    q_vec = embedding([pergunta])[0]

    sims = _cache_vectors @ q_vec / (
        np.linalg.norm(_cache_vectors, axis=1) * np.linalg.norm(q_vec)
    )

    idx = sims.argsort()[-TOP_K:][::-1]
    return [_cache_chunks[i] for i in idx]

def gerar_resposta(pergunta: str, contexto: List[str]) -> str:
    base = "\n\n---\n\n".join(contexto)

    system = (
        "Você é um assistente técnico. "
        "Use APENAS o contexto fornecido. "
        "Se não houver informação suficiente, diga 'Em geral,' e responda brevemente."
    )

    user = f"""
PERGUNTA:
{pergunta}

CONTEXTO:
{base}

RESPONDA:
- 1 parágrafo curto
- Português natural
- Termine com ponto final
"""

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_output_tokens=220,
        temperature=0.4
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
