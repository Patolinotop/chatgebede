import os, re, requests
from typing import List, Tuple
from io import BytesIO
from pypdf import PdfReader

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))

GITHUB_REPO = os.getenv("GITHUB_REPO")              # "usuario/repo"
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")            # opcional

ALLOWED_EXT = (".txt", ".pdf")

def limpar_texto(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def github_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def listar_arquivos(path: str = "") -> List[dict]:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, headers=github_headers(), timeout=25)
    if r.status_code != 200:
        return []
    data = r.json()
    out = []
    for item in data:
        if item.get("type") == "file" and item.get("name", "").lower().endswith(ALLOWED_EXT):
            out.append(item)
        elif item.get("type") == "dir":
            out.extend(listar_arquivos(item.get("path", "")))
    return out

def ler_download_url(download_url: str) -> Tuple[str, str]:
    """
    returns: (texto_limpo, tipo) onde tipo in {"txt","pdf",""}
    """
    r = requests.get(download_url, timeout=35)
    if r.status_code != 200:
        return "", ""
    if download_url.lower().endswith(".txt"):
        return limpar_texto(r.text), "txt"
    if download_url.lower().endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(r.content))
            txt = " ".join((p.extract_text() or "") for p in reader.pages)
            return limpar_texto(txt), "pdf"
        except Exception:
            return "", ""
    return "", ""

def chunkar(texto: str) -> List[str]:
    if not texto:
        return []
    chunks = []
    i = 0
    while i < len(texto):
        chunks.append(texto[i:i + CHUNK_SIZE])
        i += max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks

def carregar_documentos_github() -> List[dict]:
    """
    retorna lista de dicts:
      {"source": url, "content": chunk_text}
    """
    if not GITHUB_REPO:
        raise RuntimeError("GITHUB_REPO ausente")
    items = listar_arquivos("")
    docs = []
    for it in items:
        download_url = it.get("download_url")
        if not download_url:
            continue
        texto, _tipo = ler_download_url(download_url)
        for c in chunkar(texto):
            docs.append({"source": download_url, "content": c})
    return docs
