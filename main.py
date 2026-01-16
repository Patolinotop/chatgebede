# ================================
# MENU EB – Backend Chatbot API (RAG SEMÂNTICO COM EMBEDDINGS)
# STATUS: ATUALIZADO – MENOS REPETIÇÃO + MELHOR MATCH DE CONTEXTO
#
# PRINCIPAIS MELHORIAS:
# - Remove interseção de palavras e usa embeddings (busca semântica real)
# - Cache do índice de embeddings (evita custo/latência por request)
# - Se contexto vazio: não chama o modelo (evita "não descrito" confuso)
# - Temperature ajustada para reduzir repetição sem perder controle
# - Mantém: resposta curta, sem inventar, sem copiar frases inteiras
# ================================

from flask import Flask, request
import os, json, re, requests, traceback, time, hashlib
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# ================================
# ENV
# ================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # opcional

# Cache
CACHE_TTL_FILES = int(os.getenv("CACHE_TTL_FILES", "300"))     # 5 min
CACHE_TTL_TEXTS = int(os.getenv("CACHE_TTL_TEXTS", "300"))     # 5 min
CACHE_TTL_INDEX = int(os.getenv("CACHE_TTL_INDEX", "3600"))    # 1h (embeddings)

# Resposta
MIN_CHARS = int(os.getenv("MIN_CHARS", "120"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "160"))

# Ajustes RAG
TOP_K = int(os.getenv("TOP_K", "10"))
MIN_SIM = float(os.getenv("MIN_SIM", "0.20"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "7000"))

# Ajustes geração
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.35"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "120"))

DEBUG = os.getenv("DEBUG", "0") == "1"

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes (OPENAI_API_KEY, GITHUB_REPO)")

client = OpenAI(api_key=OPENAI_API_KEY)

# ================================
# APP
# ================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# ================================
# Utils
# ================================
def log(*args):
    if DEBUG:
        print("[DEBUG]", *args)

def limpar_texto(txt: str) -> str:
    txt = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    txt = re.sub(r"\n{2,}", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

TERMOS_FIXOS = [
    "Graduados", "Praças", "Oficiais", "Exército Brasileiro",
    "CDP", "BPE", "Promoções", "Recrutamento"
]

def aplicar_capitalizacao(texto: str) -> str:
    for termo in TERMOS_FIXOS:
        pattern = r"(?<!\w)" + re.escape(termo) + r"(?!\w)"
        texto = re.sub(pattern, termo, texto, flags=re.IGNORECASE)
    return texto

def dividir_em_trechos(texto: str, chunk_size: int = 900) -> List[str]:
    texto = limpar_texto(texto)
    if len(texto) <= chunk_size:
        return [texto]
    trechos = []
    i = 0
    while i < len(texto):
        j = min(i + chunk_size, len(texto))
        cut = texto.rfind(".", i, j)
        if cut != -1 and cut > i + 200:
            j = cut + 1
        trechos.append(texto[i:j].strip())
        i = j
    return [t for t in trechos if t]

def garantir_pontuacao_final(texto: str) -> str:
    texto = texto.strip()
    if not texto:
        return texto
    if texto[-1] not in ".!?":
        texto = re.sub(r"[,:;–—-]\s*$", "", texto).strip()
        if texto and texto[-1] not in ".!?":
            texto += "."
    return texto

def contar_chars(texto: str) -> int:
    return len(texto)

def resposta_parece_cortada(texto: str) -> bool:
    t = texto.strip()
    if not t:
        return True
    if re.search(r"[,;:–—-]\s*$", t):
        return True
    ultima = re.sub(r"[^\wà-ú]+$", "", t.lower()).split()[-1] if t.split() else ""
    pendentes = {
        "e", "ou", "para", "por", "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
        "em", "ao", "aos", "à", "às", "com", "sem", "sobre", "entre", "que"
    }
    if ultima in pendentes:
        return True
    if t[-1] not in ".!?" and len(t) >= max(90, MIN_CHARS - 10):
        return True
    return False

def dentro_da_faixa(texto: str, min_c: int, max_c: int) -> bool:
    n = contar_chars(texto)
    return (min_c <= n <= max_c)

# ================================
# GitHub – leitura dos .txt (com cache)
# ================================
_cache_files: Dict[str, object] = {"ts": 0.0, "items": []}
_cache_texts: Dict[str, object] = {"ts": 0.0, "items": []}

def _github_headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

def listar_txt(path: str = "") -> List[str]:
    now = time.time()
    if path == "" and _cache_files["items"] and (now - float(_cache_files["ts"]) < CACHE_TTL_FILES):
        return list(_cache_files["items"])

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, headers=_github_headers(), timeout=20)
    if r.status_code != 200:
        log("GitHub API status", r.status_code, "path=", path)
        return []

    arquivos: List[str] = []
    data = r.json()
    if isinstance(data, dict) and data.get("type") == "file":
        # caso raro: path aponta direto para um arquivo
        if data.get("name", "").endswith(".txt") and data.get("download_url"):
            arquivos.append(data["download_url"])
        return arquivos

    for item in data:
        if item.get("type") == "file" and item.get("name", "").endswith(".txt"):
            if item.get("download_url"):
                arquivos.append(item["download_url"])
        elif item.get("type") == "dir" and item.get("path"):
            arquivos.extend(listar_txt(item["path"]))

    if path == "":
        _cache_files["ts"] = now
        _cache_files["items"] = list(arquivos)

    return arquivos

def ler_txts() -> List[str]:
    now = time.time()
    if _cache_texts["items"] and (now - float(_cache_texts["ts"]) < CACHE_TTL_TEXTS):
        return list(_cache_texts["items"])

    textos: List[str] = []
    urls = listar_txt()
    log("Total .txt URLs:", len(urls))

    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text:
                textos.append(limpar_texto(r.text))
        except Exception:
            pass

    _cache_texts["ts"] = now
    _cache_texts["items"] = list(textos)
    log("Total textos carregados:", len(textos))
    return textos

# ================================
# Embeddings Index (cacheado)
# ================================
_cache_index: Dict[str, object] = {
    "ts": 0.0,
    "sig": "",
    "chunks": [],     # List[str]
    "emb": None       # np.ndarray [N, D]
}

def _signature_texts(textos: List[str]) -> str:
    # assinatura leve para detectar mudança (sem guardar tudo)
    h = hashlib.sha1()
    for t in textos:
        h.update(t[:2000].encode("utf-8", "ignore"))  # amostra por arquivo
        h.update(b"\n---\n")
    return h.hexdigest()

def _embed_texts(texts: List[str], batch_size: int = 96) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

def _cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = (m @ q)
    idx = np.argsort(-sims)[:k]
    return idx.tolist(), [float(sims[i]) for i in idx]

def build_or_get_index() -> Tuple[List[str], np.ndarray]:
    now = time.time()
    textos = ler_txts()
    sig = _signature_texts(textos)

    if (
        _cache_index["emb"] is not None
        and _cache_index["chunks"]
        and (now - float(_cache_index["ts"]) < CACHE_TTL_INDEX)
        and _cache_index["sig"] == sig
    ):
        return _cache_index["chunks"], _cache_index["emb"]

    chunks: List[str] = []
    for t in textos:
        chunks.extend(dividir_em_trechos(t, chunk_size=900))

    chunks = [c for c in chunks if c and len(c) >= 50]
    if not chunks:
        _cache_index["ts"] = now
        _cache_index["sig"] = sig
        _cache_index["chunks"] = []
        _cache_index["emb"] = np.zeros((0, 1), dtype=np.float32)
        return _cache_index["chunks"], _cache_index["emb"]

    log("Construindo embeddings para chunks:", len(chunks))
    emb = _embed_texts(chunks)

    _cache_index["ts"] = now
    _cache_index["sig"] = sig
    _cache_index["chunks"] = chunks
    _cache_index["emb"] = emb
    return chunks, emb

def montar_contexto_relevante_semantico(tema: str, max_chars: int = 7000, top_k: int = 10) -> str:
    chunks, emb = build_or_get_index()
    if emb is None or emb.shape[0] == 0:
        return ""

    qv = _embed_texts([tema])[0]
    idxs, sims = _cosine_topk(qv, emb, k=min(top_k, emb.shape[0]))

    selecionados: List[str] = []
    total = 0

    for i, s in zip(idxs, sims):
        if s < MIN_SIM:
            continue
        bloco = chunks[i].strip()
        add_len = len(bloco) + 8
        if total + add_len > max_chars:
            break
        selecionados.append(bloco)
        total += add_len

    contexto = "\n\n---\n\n".join(selecionados)
    log("Contexto semântico chars:", len(contexto), "top sims:", sims[:5])
    return contexto

# ================================
# OpenAI – geração controlada
# ================================
def _call_openai(system_prompt: str, user_prompt: str, max_output_tokens: int) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_output_tokens=max_output_tokens,
    )
    return (resp.output_text or "").strip()

def gerar_resposta(tema: str, contexto: str) -> Tuple[Optional[str], Optional[str]]:
    """
    1) Gera resposta curta diretamente.
    2) Se vier fora da faixa ou "cortada", pede reescrita curta (2ª chamada).
    """
    system_prompt = (
        "Você é um analista de normas institucionais. "
        "Use EXCLUSIVAMENTE as informações contidas nos DOCUMENTOS fornecidos. "
        "É PROIBIDO completar lacunas com conhecimento externo, suposições ou práticas comuns. "
        "Se um detalhe não estiver explícito, OMITA esse detalhe. "
        "Reescreva com palavras próprias, sem copiar frases inteiras. "
        "Varie a redação quando possível (sinônimos e ordem), sem mudar o conteúdo factual. "
        "Entregue texto curto, gramaticalmente correto, natural e finalizado (sem frase cortada)."
    )

    user_prompt = (
        f"TEMA/PERGUNTA: {tema}\n\n"
        "DOCUMENTOS (TRECHOS RELEVANTES):\n"
        f"{contexto}\n\n"
        "REGRAS DE RESPOSTA (OBRIGATÓRIO):\n"
        "1) Responda APENAS sobre o tema.\n"
        "2) Use SOMENTE o que está nos documentos (sem inferir).\n"
        "3) Não copie frases inteiras; reescreva.\n"
        f"4) Produza 1 único parágrafo com {MIN_CHARS} a {MAX_CHARS} caracteres (contando espaços).\n"
        "5) Termine com ponto final.\n"
        "6) Se o tema for amplo, foque no conceito central presente nos documentos.\n"
    )

    try:
        texto = _call_openai(system_prompt, user_prompt, max_output_tokens=MAX_OUTPUT_TOKENS)
        if not texto:
            raise RuntimeError("Resposta vazia da OpenAI")

        texto = aplicar_capitalizacao(texto)
        texto = garantir_pontuacao_final(texto)

        if (not dentro_da_faixa(texto, MIN_CHARS, MAX_CHARS)) or resposta_parece_cortada(texto):
            rewrite_prompt = (
                f"TEMA/PERGUNTA: {tema}\n\n"
                "DOCUMENTOS (TRECHOS RELEVANTES):\n"
                f"{contexto}\n\n"
                "TEXTO ATUAL (NÃO CONFIE NELE SE ESTIVER LONGO/CORTADO):\n"
                f"{texto}\n\n"
                "TAREFA:\n"
                f"- REESCREVA para ficar ENTRE {MIN_CHARS} e {MAX_CHARS} caracteres (com espaços).\n"
                "- Mantenha SOMENTE informações explícitas nos documentos.\n"
                "- 1 único parágrafo, formal, objetivo e natural.\n"
                "- Termine com ponto final.\n"
                "- Não deixe frase incompleta.\n"
            )
            texto2 = _call_openai(system_prompt, rewrite_prompt, max_output_tokens=MAX_OUTPUT_TOKENS).strip()
            if texto2:
                texto2 = aplicar_capitalizacao(texto2)
                texto2 = garantir_pontuacao_final(texto2)

                if dentro_da_faixa(texto2, MIN_CHARS, MAX_CHARS) and not resposta_parece_cortada(texto2):
                    texto = texto2
                else:
                    texto = re.sub(r"\s{2,}", " ", texto2 if texto2 else texto).strip()
                    texto = garantir_pontuacao_final(texto)

        if (not dentro_da_faixa(texto, MIN_CHARS, MAX_CHARS)) or resposta_parece_cortada(texto):
            texto = (
                "Os documentos disponíveis não trazem informação explícita suficiente sobre o tema solicitado para afirmar regras ou detalhes adicionais."
            )
            texto = aplicar_capitalizacao(texto)
            texto = garantir_pontuacao_final(texto)

            if not dentro_da_faixa(texto, MIN_CHARS, MAX_CHARS):
                fallback_prompt = (
                    f"Reescreva o texto a seguir para ficar ENTRE {MIN_CHARS} e {MAX_CHARS} caracteres (com espaços), "
                    "1 parágrafo formal, natural e terminando com ponto:\n"
                    f"{texto}"
                )
                texto_fb = _call_openai(
                    "Você é um redator técnico. Mantenha o sentido, sem adicionar fatos novos.",
                    fallback_prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS
                ).strip()
                if texto_fb:
                    texto_fb = garantir_pontuacao_final(texto_fb)
                    if dentro_da_faixa(texto_fb, MIN_CHARS, MAX_CHARS) and not resposta_parece_cortada(texto_fb):
                        texto = texto_fb

        return texto, None

    except Exception as e:
        traceback.print_exc()
        return None, str(e)

# ================================
# API
# ================================
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = (data.get("input", "") or "").strip()

    if not tema:
        return app.response_class(
            response=json.dumps({"error": "tema_vazio"}, ensure_ascii=False),
            mimetype="application/json"
        )

    # Contexto semântico (embeddings)
    contexto = montar_contexto_relevante_semantico(tema, max_chars=CONTEXT_MAX_CHARS, top_k=TOP_K)

    # Se não há base, não chama o modelo (evita negar de forma "errada")
    if not contexto.strip():
        return app.response_class(
            response=json.dumps({
                "reply": "Não há trechos suficientes nas fontes para responder com segurança sobre esse tema."
            }, ensure_ascii=False),
            mimetype="application/json"
        )

    texto, erro = gerar_resposta(tema, contexto)

    if erro:
        return app.response_class(
            response=json.dumps({"error": "openai_failed", "detail": erro}, ensure_ascii=False),
            mimetype="application/json"
        )

    return app.response_class(
        response=json.dumps({"reply": texto}, ensure_ascii=False),
        mimetype="application/json"
    )

# ================================
# START
# ================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
