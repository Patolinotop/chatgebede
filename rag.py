import os
import re
import time
import requests
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from io import BytesIO

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Modelos
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")

# Chunking / busca
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Relevância mínima (pode ajustar)
MIN_SIM = float(os.getenv("MIN_SIM", "0.22"))

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes: OPENAI_API_KEY e/ou GITHUB_REPO")

client = OpenAI(api_key=OPENAI_API_KEY)

SPACE_RE = re.compile(r"\s+")

def limpar_texto(t: str) -> str:
    t = t or ""
    t = t.replace("\u200b", "")
    t = SPACE_RE.sub(" ", t).strip()
    return t

def github_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def listar_arquivos(path: str = "") -> List[str]:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, headers=github_headers(), timeout=20)
    if r.status_code != 200:
        return []

    data = r.json()
    arquivos: List[str] = []

    for item in data:
        if item.get("type") == "file":
            name = (item.get("name") or "").lower()
            if name.endswith((".txt", ".pdf")):
                dl = item.get("download_url")
                if dl:
                    arquivos.append(dl)
        elif item.get("type") == "dir":
            p = item.get("path")
            if p:
                arquivos.extend(listar_arquivos(p))

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
            full = " ".join((p.extract_text() or "") for p in reader.pages)
            return limpar_texto(full)
        except Exception:
            return ""

    return ""

def chunkar(texto: str) -> List[str]:
    texto = texto or ""
    if not texto:
        return []
    chunks = []
    i = 0
    while i < len(texto):
        chunks.append(texto[i:i + CHUNK_SIZE])
        i += max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

# Cache em memória
_chunks: List[str] = []
_vectors: Optional[np.ndarray] = None
_base_carregada = False
_last_load_info: Dict[str, Any] = {}

def carregar_base() -> None:
    global _chunks, _vectors, _base_carregada, _last_load_info

    if _base_carregada:
        return

    t0 = time.time()
    print("[RAG] Carregando base do GitHub...")

    arquivos = listar_arquivos()
    textos: List[str] = []

    for url in arquivos:
        texto = ler_arquivo(url)
        if not texto:
            continue
        for c in chunkar(texto):
            textos.append(f"FONTE: {url}\n{c}")

    if not textos:
        print("[RAG] Nenhum documento encontrado para indexar.")
        _chunks = []
        _vectors = None
        _base_carregada = True
        _last_load_info = {"chunks": 0, "seconds": time.time() - t0}
        return

    print(f"[RAG] Gerando embeddings de {len(textos)} chunks...")
    resp = client.embeddings.create(model=EMBED_MODEL, input=textos)

    _chunks = textos
    _vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    _base_carregada = True

    dt = time.time() - t0
    _last_load_info = {"chunks": len(_chunks), "seconds": dt}
    print(f"[RAG] Base carregada. chunks={len(_chunks)} em {dt:.2f}s")

def _cosine_sims(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    # matrix: (n, d), vec: (d,)
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec))
    denom = np.where(denom == 0.0, 1e-12, denom)
    return (matrix @ vec) / denom

def search_context(pergunta: str) -> List[str]:
    carregar_base()
    if not _chunks or _vectors is None:
        return []

    pergunta = limpar_texto(pergunta)
    if not pergunta:
        return []

    q_emb = client.embeddings.create(model=EMBED_MODEL, input=[pergunta]).data[0].embedding
    q_vec = np.array(q_emb, dtype=np.float32)

    sims = _cosine_sims(_vectors, q_vec)
    if sims.size == 0:
        return []

    idx = sims.argsort()[-TOP_K:][::-1]
    return [_chunks[int(i)] for i in idx]

def context_is_relevant(ctx: Any) -> Dict[str, Any]:
    """
    Sempre devolve dict:
      { "relevant": bool, "best_sim": float }

    - Se ctx vier como (best_sim, ctx_list) ou (ctx_list, best_sim), tenta entender.
    - Se ctx vier só como lista, marca best_sim como 0.0 (sem sinal numérico).
    """
    # Caso venha como tupla (algumas versões antigas faziam isso)
    if isinstance(ctx, tuple) and len(ctx) == 2:
        a, b = ctx

        # a é sim?
        if isinstance(a, (int, float)) and isinstance(b, list):
            best_sim = float(a)
            return {"relevant": best_sim >= MIN_SIM, "best_sim": best_sim}

        # b é sim?
        if isinstance(b, (int, float)) and isinstance(a, list):
            best_sim = float(b)
            return {"relevant": best_sim >= MIN_SIM, "best_sim": best_sim}

        # se veio ( (sim,idx), ctx_list ) ou coisa estranha
        if isinstance(a, (list, tuple)) and len(a) >= 1 and isinstance(a[0], (int, float)):
            best_sim = float(a[0])
            return {"relevant": best_sim >= MIN_SIM, "best_sim": best_sim}

        return {"relevant": False, "best_sim": 0.0}

    # Caso normal: ctx é lista de chunks
    if isinstance(ctx, list):
        # Não temos best_sim numérico aqui, então retornamos 0.0 (o bot só usa como “sinal”).
        # Se você quiser, eu adapto o search_context para também retornar o best_sim real.
        return {"relevant": True if len(ctx) > 0 else False, "best_sim": 0.0}

    return {"relevant": False, "best_sim": 0.0}
