# ================================
# MENU EB – Backend Chatbot API (ANÁLISE SEMÂNTICA CONTROLADA)
# STATUS: ATUALIZADO – MAIS ANCORADO, MENOS ALUCINAÇÃO
#
# PRINCIPAIS AJUSTES (SEM MUDAR A LÓGICA DO PROJETO):
# - Troca de modelo default para melhor custo/benefício (gpt-4o-mini)
# - Contexto não é mais "bloco amorfo": agora é RECORTADO por RELEVÂNCIA ao tema
# - Prompt mais rígido contra inferência externa + instrução de "não preencher lacunas"
# - Temperatura baixa
# - Validação de comprimento (120–160 caracteres) e fallback seguro
# - Cache simples (lista de .txt e conteúdo) para estabilidade e performance
# - Melhorias de robustez: rate-limit GitHub (token opcional), timeouts, logs
# ================================

from flask import Flask, request
import os, json, re, requests, traceback, time
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from openai import OpenAI

# ================================
# ENV
# ================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # bom custo/benefício :contentReference[oaicite:1]{index=1}

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # opcional (evita rate limit)

# Cache (segundos)
CACHE_TTL_FILES = int(os.getenv("CACHE_TTL_FILES", "300"))     # 5 min
CACHE_TTL_TEXTS = int(os.getenv("CACHE_TTL_TEXTS", "300"))     # 5 min

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
    # Mais robusto para termos com espaços/siglas
    for termo in TERMOS_FIXOS:
        pattern = r"(?<!\w)" + re.escape(termo) + r"(?!\w)"
        texto = re.sub(pattern, termo, texto, flags=re.IGNORECASE)
    return texto

def normalizar_palavras(s: str) -> List[str]:
    # tokens simples (sem embeddings) pra selecionar trechos relevantes
    s = s.lower()
    s = re.sub(r"[^a-zà-ú0-9\s]", " ", s, flags=re.IGNORECASE)
    parts = [p for p in s.split() if len(p) >= 3]
    # remove alguns termos muito genéricos
    stop = {"para", "com", "sem", "sobre", "como", "qual", "quais", "porque", "porquê",
            "uma", "uns", "umas", "dos", "das", "que", "não", "nos", "nas", "entre"}
    return [p for p in parts if p not in stop]

def pontuar_relevancia(tema: str, trecho: str) -> int:
    # score bem simples (mas já reduz alucinação por “mistura geral”)
    t = set(normalizar_palavras(tema))
    if not t:
        return 0
    x = set(normalizar_palavras(trecho))
    return len(t.intersection(x))

def dividir_em_trechos(texto: str, chunk_size: int = 900) -> List[str]:
    # quebra por tamanho, tentando respeitar pontos finais
    texto = limpar_texto(texto)
    if len(texto) <= chunk_size:
        return [texto]
    trechos = []
    i = 0
    while i < len(texto):
        j = min(i + chunk_size, len(texto))
        # tenta cortar em ponto final para evitar pedaços “quebrados”
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
        texto = texto.rsplit(" ", 1)[0] + "."
    return texto

def ajustar_tamanho(texto: str, min_c: int = 120, max_c: int = 160) -> str:
    texto = texto.strip()
    if len(texto) > max_c:
        # corta no último espaço antes do limite e pontua
        texto = texto[:max_c]
        texto = texto.rsplit(" ", 1)[0].strip()
        texto = garantir_pontuacao_final(texto)
    elif len(texto) < min_c:
        # não inventa pra “encher”: mantém seguro e formal
        # (o prompt também tenta segurar isso)
        texto = garantir_pontuacao_final(texto)
    return texto

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
    # cache da lista
    now = time.time()
    if _cache_files["items"] and (now - float(_cache_files["ts"]) < CACHE_TTL_FILES) and path == "":
        return list(_cache_files["items"])

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, headers=_github_headers(), timeout=15)
    if r.status_code != 200:
        print("[DEBUG] GitHub API status", r.status_code, "path=", path)
        return []

    arquivos: List[str] = []
    for item in r.json():
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
    # cache do conteúdo
    now = time.time()
    if _cache_texts["items"] and (now - float(_cache_texts["ts"]) < CACHE_TTL_TEXTS):
        return list(_cache_texts["items"])

    textos: List[str] = []
    for url in listar_txt():
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.text:
                textos.append(limpar_texto(r.text))
        except Exception:
            pass

    _cache_texts["ts"] = now
    _cache_texts["items"] = list(textos)
    return textos

def montar_contexto_relevante(tema: str, max_chars: int = 7000) -> str:
    """
    Mantém a lógica (txts como base), mas evita mandar tudo misturado.
    Seleciona trechos mais relevantes ao tema por sobreposição de palavras.
    """
    textos = ler_txts()
    trechos: List[Tuple[int, str]] = []

    for t in textos:
        for ch in dividir_em_trechos(t, chunk_size=900):
            score = pontuar_relevancia(tema, ch)
            if score > 0:
                trechos.append((score, ch))

    # Se o tema for muito vago e não bater nada, pega poucos trechos iniciais
    # (melhor do que "bloco amorfo" e ajuda perguntas genéricas)
    if not trechos and textos:
        fallback = []
        for t in textos[:3]:
            fallback.extend(dividir_em_trechos(t, chunk_size=900)[:1])
        contexto = "\n\n---\n\n".join(fallback)
        return contexto[:max_chars]

    trechos.sort(key=lambda x: x[0], reverse=True)

    selecionados: List[str] = []
    total = 0
    for score, ch in trechos:
        bloco = ch.strip()
        if not bloco:
            continue
        add_len = len(bloco) + 8
        if total + add_len > max_chars:
            break
        selecionados.append(bloco)
        total += add_len

    contexto = "\n\n---\n\n".join(selecionados)
    print("[DEBUG] Contexto relevante chars:", len(contexto))
    return contexto

# ================================
# OpenAI – RACIOCÍNIO CONTROLADO (mais rígido)
# ================================
def gerar_resposta(tema: str, contexto: str) -> Tuple[str, str]:
    """
    Retorna (texto, erro). Texto deve ser 120–160 caracteres e baseado somente no contexto.
    """
    system_prompt = (
        "Você é um analista de normas institucionais. "
        "Use EXCLUSIVAMENTE as informações contidas nos DOCUMENTOS fornecidos. "
        "É PROIBIDO completar lacunas com conhecimento externo, suposições ou práticas comuns. "
        "Se um detalhe não estiver explícito nos documentos, você deve OMITIR esse detalhe. "
        "Se os documentos não trouxerem base suficiente para responder ao tema, "
        "responda de forma formal informando que os documentos não especificam o ponto, sem inventar."
    )

    user_prompt = (
        f"TEMA/PERGUNTA: {tema}\n\n"
        "DOCUMENTOS (TRECHOS RELEVANTES):\n"
        f"{contexto}\n\n"
        "REGRAS DE RESPOSTA (OBRIGATÓRIO):\n"
        "1) Responda APENAS sobre o tema/pergunta.\n"
        "2) Use SOMENTE o que está nos documentos (sem inferir, sem generalizar, sem 'completar').\n"
        "3) Não copie frases inteiras dos documentos; reescreva com suas palavras.\n"
        "4) Produza 1 único parágrafo com 120 a 160 caracteres (contando espaços).\n"
        "5) Linguagem formal, humana e objetiva.\n"
        "6) Se o tema for amplo, foque no conceito central que aparece nos documentos.\n"
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,            # reduz criatividade/alucinação
            max_output_tokens=120       # suficiente para 120–160 caracteres em PT
        )

        texto = (response.output_text or "").strip()
        if not texto:
            raise RuntimeError("Resposta vazia da OpenAI")

        texto = aplicar_capitalizacao(texto)
        texto = ajustar_tamanho(texto, 120, 160)
        texto = garantir_pontuacao_final(texto)

        # Garantia extra: se ainda sair MUITO fora da faixa, aplica fallback seguro
        if len(texto) < 80 or len(texto) > 220:
            texto = "Os documentos disponíveis não especificam detalhes suficientes sobre o tema solicitado, sem base para afirmar regras adicionais."
            texto = ajustar_tamanho(texto, 120, 160)
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

    # Contexto por relevância (evita misturar assuntos e “inventar ponte”)
    contexto = montar_contexto_relevante(tema, max_chars=7000)

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
