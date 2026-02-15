import os, time, traceback
import numpy as np
from typing import List, Tuple
from openai import OpenAI
from kb_github import carregar_documentos_github

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIM = float(os.getenv("MIN_SIM", "0.25"))

RELOAD_SECONDS = int(os.getenv("RELOAD_SECONDS", "3600"))
MAX_DOCS = int(os.getenv("MAX_DOCS", "5000"))  # segurança pra não explodir custo/memória

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_chunks: List[dict] = []
_vectors = None
_loaded = False
_last_load_ts = 0.0

def _log(msg: str):
    print(f"[RAG] {msg}", flush=True)

def _embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def ensure_loaded(force: bool = False):
    global _chunks, _vectors, _loaded, _last_load_ts
    now = time.time()
    if _loaded and not force and (now - _last_load_ts) < RELOAD_SECONDS:
        return

    _log("Carregando base do GitHub...")
    t0 = time.time()
    try:
        docs = carregar_documentos_github()
        if len(docs) > MAX_DOCS:
            docs = docs[:MAX_DOCS]
            _log(f"Limitando docs/chunks a MAX_DOCS={MAX_DOCS} (evitar custo/memória).")

        if not docs:
            _chunks = []
            _vectors = None
            _loaded = True
            _last_load_ts = now
            _log("Nenhum documento encontrado no GitHub.")
            return

        texts = [f"FONTE: {d['source']}\n{d['content']}" for d in docs]
        _log(f"Gerando embeddings de {len(texts)} chunks...")
        vecs = _embed_texts(texts)

        _chunks = [{"source": d["source"], "text": texts[i]} for i, d in enumerate(docs)]
        _vectors = vecs
        _loaded = True
        _last_load_ts = now
        _log(f"Base carregada. chunks={len(_chunks)} em {time.time()-t0:.2f}s")

    except Exception:
        _log("ERRO carregando base:")
        traceback.print_exc()
        # não marca como loaded pra tentar depois
        _chunks = []
        _vectors = None
        _loaded = False

def search_context(query: str) -> Tuple[List[str], float]:
    ensure_loaded()
    if not _chunks or _vectors is None:
        return [], 0.0

    try:
        q_vec = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
        q = np.array(q_vec, dtype=np.float32)

        denom = (np.linalg.norm(_vectors, axis=1) * (np.linalg.norm(q) + 1e-8) + 1e-8)
        sims = (_vectors @ q) / denom

        idx = sims.argsort()[-TOP_K:][::-1]
        best = float(sims[idx[0]]) if len(idx) else 0.0
        contexts = [_chunks[i]["text"] for i in idx]
        return contexts, best
    except Exception:
        _log("ERRO ao buscar contexto:")
        traceback.print_exc()
        return [], 0.0

def context_is_relevant(best_sim: float) -> bool:
    return best_sim >= MIN_SIM
