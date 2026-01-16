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
CACHE_TTL_FILES = int(os.getenv("CACHE_TTL_FILES", "300"))      # 5 min
CACHE_TTL_TEXTS = int(os.getenv("CACHE_TTL_TEXTS", "300"))      # 5 min
CACHE_TTL_INDEX = int(os.getenv("CACHE_TTL_INDEX", "3600"))     # 1h

# Resposta
MIN_CHARS = int(os.getenv("MIN_CHARS", "120"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "160"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "220"))

# Geração (menos repetição)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.55"))

# RAG
TOP_K = int(os.getenv("TOP_K", "35"))
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "14000"))

# Score híbrido
MIN_SIM = float(os.getenv("MIN_SIM", "0.07"))
LEX_BOOST_PER_HIT = float(os.getenv("LEX_BOOST_PER_HIT", "0.035"))
LEX_BOOST_CAP = float(os.getenv("LEX_BOOST_CAP", "0.18"))

# Permitir conhecimento geral quando faltar algo nos txts
ALLOW_GENERAL_KNOWLEDGE = os.getenv("ALLOW_GENERAL_KNOWLEDGE", "1") == "1"

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

def resposta_parece_cortada(texto: str) -> bool:
    t = texto.strip()
    if not t:
        return True
    if re.search(r"[,;:–—-]\s*$", t):
        return True
    ultima = re.sub(r"[^\wà-ú]+$", "", t.lower()).split()[-1] if t.split() else ""
    pendentes = {"e","ou","para","por","de","do","da","dos","das","no","na","nos","nas","em","ao","aos","à","às","com","sem","sobre","entre","que"}
    if ultima in pendentes:
        return True
    if t[-1] not in ".!?" and len(t) >= max(90, MIN_CHARS - 10):
        return True
    return False

def dentro_da_faixa(texto: str, min_c: int, max_c: int) -> bool:
    n = len(texto)
    return min_c <= n <= max_c

def normalizar_palavras(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-zà-ú0-9\s/]", " ", s, flags=re.IGNORECASE)
    parts = [p for p in s.split() if len(p) >= 2]
    stop = {
        "para","com","sem","sobre","como","qual","quais","porque","porquê",
        "uma","uns","umas","dos","das","que","não","nos","nas","entre",
        "sua","seu","suas","seus","esse","essa","isso","aquele","aquela",
        "importância","importante","sobre","tema","assunto","do","da","de"
    }
    return [p for p in parts if p not in stop]


# ================================
# Expansão de tema (ajuda MUITO)
# ================================
EXPAND_MAP = {
    "constituição": ["Constituição Federal", "CF", "CF/88", "legalidade", "direitos", "deveres", "normas"],
    "graduados": ["graduado", "sargento", "sargentos", "subtenente", "subtenentes"],
    "praças": ["praça", "soldado", "cabo", "sargento"],
    "recrutamento": ["alistamento", "ingresso", "incorporação", "seleção"],
    "promoções": ["promoção", "progressão", "ascensão", "antiguidade", "merecimento"],
}

def expandir_tema(tema: str) -> Tuple[List[str], List[str]]:
    t = tema.strip()
    lower = t.lower()

    phrases = [t]
    lex_terms = normalizar_palavras(t)

    for k, vals in EXPAND_MAP.items():
        if k in lower:
            phrases.extend([f"{t} {v}" for v in vals])
            lex_terms.extend(normalizar_palavras(" ".join(vals)))

    # remove duplicados mantendo ordem
    seen = set()
    lex_unique = []
    for w in lex_terms:
        if w not in seen:
            seen.add(w)
            lex_unique.append(w)

    return phrases[:6], lex_unique[:24]


# ================================
# GitHub – leitura dos .txt (cache)
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
        log("GitHub status", r.status_code, "path=", path)
        return []

    data = r.json()
    arquivos: List[str] = []

    if isinstance(data, dict) and data.get("type") == "file":
        if data.get("name", "").endswith(".txt") and data.get("download_url"):
            return [data["download_url"]]
        return []

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
    log("txt urls:", len(urls))

    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text:
                textos.append(limpar_texto(r.text))
        except Exception:
            pass

    _cache_texts["ts"] = now
    _cache_texts["items"] = list(textos)
    log("txt carregados:", len(textos))
    return textos


# ================================
# Índice de embeddings (cache)
# ================================
_cache_index: Dict[str, object] = {"ts": 0.0, "sig": "", "chunks": [], "emb": None}

def _signature_texts(textos: List[str]) -> str:
    h = hashlib.sha1()
    for t in textos:
        h.update(str(len(t)).encode("utf-8"))
        h.update(t[:2500].encode("utf-8", "ignore"))
        h.update(t[-800:].encode("utf-8", "ignore"))
        h.update(b"\n--\n")
    return h.hexdigest()

def _embed_texts(texts: List[str], batch_size: int = 96) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

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

    log("chunks:", len(chunks), "criando embeddings...")
    emb = _embed_texts(chunks)

    _cache_index["ts"] = now
    _cache_index["sig"] = sig
    _cache_index["chunks"] = chunks
    _cache_index["emb"] = emb
    return chunks, emb

def _cosine(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return (m @ q)

def _lex_boost(chunk: str, lex_terms: List[str]) -> float:
    if not lex_terms:
        return 0.0
    text = chunk.lower()
    hits = 0
    for w in lex_terms:
        if w in text:
            hits += 1
    boost = min(LEX_BOOST_CAP, hits * LEX_BOOST_PER_HIT)
    return float(boost)

def montar_contexto_hibrido(tema: str, max_chars: int, top_k: int) -> Tuple[str, Dict[str, object]]:
    """
    Retorna:
      - contexto (string com trechos)
      - meta (info debug)
    """
    phrases, lex_terms = expandir_tema(tema)
    chunks, emb = build_or_get_index()
    if emb is None or emb.shape[0] == 0:
        return "", {"reason": "no_chunks"}

    # embedding do tema (usa a primeira phrase como query principal)
    qv = _embed_texts([phrases[0]])[0]
    sims = _cosine(qv, emb)

    # score híbrido = cosine + boost lexical
    scores = []
    for i in range(len(chunks)):
        s = float(sims[i])
        s += _lex_boost(chunks[i], lex_terms)
        scores.append(s)

    idxs = np.argsort(-np.array(scores))[: min(top_k, len(scores))].tolist()
    best_scores = [float(scores[i]) for i in idxs[:10]]

    # seleciona trechos: primeiro os acima de MIN_SIM, senão pega top mesmo assim
    selecionados = []
    total = 0

    passed = 0
    for i in idxs:
        if scores[i] < MIN_SIM:
            continue
        bloco = chunks[i].strip()
        add_len = len(bloco) + 8
        if total + add_len > max_chars:
            break
        selecionados.append(bloco)
        total += add_len
        passed += 1

    if not selecionados:
        # fallback: pega top trechos mesmo (evita contexto vazio)
        total = 0
        for i in idxs[: max(10, top_k // 2)]:
            bloco = chunks[i].strip()
            add_len = len(bloco) + 8
            if total + add_len > max_chars:
                break
            selecionados.append(bloco)
            total += add_len

    contexto = "\n\n---\n\n".join(selecionados)
    meta = {
        "expanded_phrases": phrases,
        "lex_terms": lex_terms[:10],
        "top_scores": best_scores,
        "passed_min_sim": passed,
        "context_chars": len(contexto),
    }
    log("meta:", meta)
    return contexto, meta


# ================================
# OpenAI – geração (RAG + conhecimento geral permitido)
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
    Regras:
    - Sempre usar o contexto como base do "universo" (Graduados etc.)
    - Pode usar conhecimento geral (ex: Constituição) para conectar,
      mas SEM dizer que veio dos trechos.
    - Deve caber em 120–160 chars, 1 parágrafo e ponto final.
    """

    if ALLOW_GENERAL_KNOWLEDGE:
        policy = (
            "Você pode complementar com conhecimento geral APENAS quando o assunto não aparecer nos trechos. "
            "Quando usar conhecimento geral, sinalize com termos como 'Em geral,' ou 'De modo geral,' "
            "e NÃO afirme que isso consta nos documentos. "
            "NUNCA contradiga os trechos."
        )
    else:
        policy = (
            "Use SOMENTE o que está nos trechos. Não use conhecimento externo."
        )

    system_prompt = (
        "Você é um redator técnico, claro e humano. "
        "Escreva em português correto, natural, sem soar robótico. "
        + policy
    )

    user_prompt = (
        f"TEMA: {tema}\n\n"
        "TRECHOS DA BASE (USE COMO FUNDAMENTO):\n"
        f"{contexto}\n\n"
        "TAREFA:\n"
        f"- Escreva 1 único parágrafo com {MIN_CHARS} a {MAX_CHARS} caracteres (com espaços).\n"
        "- Baseie a parte principal nos trechos (reformule com suas palavras, sem copiar frases).\n"
        "- Se precisar de conexão geral (ex: Constituição), faça uma frase curta do tipo 'Em geral,...' sem fingir que está nos trechos.\n"
        "- Termine com ponto final.\n"
    )

    try:
        texto = _call_openai(system_prompt, user_prompt, max_output_tokens=MAX_OUTPUT_TOKENS)
        if not texto:
            raise RuntimeError("Resposta vazia da OpenAI")

        texto = aplicar_capitalizacao(texto)
        texto = garantir_pontuacao_final(texto)

        # 2ª tentativa se ficou fora da faixa ou truncada
        if (not dentro_da_faixa(texto, MIN_CHARS, MAX_CHARS)) or resposta_parece_cortada(texto):
            rewrite_prompt = (
                f"TEMA: {tema}\n\n"
                "TRECHOS DA BASE:\n"
                f"{contexto}\n\n"
                "TEXTO ATUAL:\n"
                f"{texto}\n\n"
                "REESCREVA:\n"
                f"- Entre {MIN_CHARS} e {MAX_CHARS} caracteres.\n"
                "- 1 parágrafo, natural, objetivo.\n"
                "- Use trechos como base; se usar geral, diga 'Em geral,' e não atribua aos trechos.\n"
                "- Termine com ponto final.\n"
            )
            texto2 = _call_openai(system_prompt, rewrite_prompt, max_output_tokens=MAX_OUTPUT_TOKENS).strip()
            if texto2:
                texto2 = aplicar_capitalizacao(texto2)
                texto2 = garantir_pontuacao_final(texto2)
                if dentro_da_faixa(texto2, MIN_CHARS, MAX_CHARS) and not resposta_parece_cortada(texto2):
                    texto = texto2

        # fallback final: ainda assim responde, mas sem inventar detalhe específico
        if (not dentro_da_faixa(texto, MIN_CHARS, MAX_CHARS)) or resposta_parece_cortada(texto):
            if ALLOW_GENERAL_KNOWLEDGE:
                texto = "Nos trechos, Graduados aparecem ligados a funções e deveres. Em geral, a Constituição orienta direitos e legalidade."
            else:
                texto = "Os trechos recuperados não trazem base suficiente para resumir o tema com segurança, sem adicionar informação externa."
            texto = aplicar_capitalizacao(texto)
            texto = garantir_pontuacao_final(texto)

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

    contexto, meta = montar_contexto_hibrido(
        tema,
        max_chars=CONTEXT_MAX_CHARS,
        top_k=TOP_K
    )

    if not contexto.strip():
        # mesmo aqui, se conhecimento geral for permitido, dá uma saída curta
        if ALLOW_GENERAL_KNOWLEDGE:
            reply = "Não encontrei trechos na base para sustentar o tema. Em geral, o assunto depende de normas e deveres aplicáveis."
        else:
            reply = "Não foi possível recuperar trechos da base para sustentar a resposta com segurança."
        reply = aplicar_capitalizacao(garantir_pontuacao_final(reply))
        return app.response_class(
            response=json.dumps({"reply": reply}, ensure_ascii=False),
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
