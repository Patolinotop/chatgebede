from flask import Flask, request
import os, json, re, requests, traceback, time, hashlib
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # opcional

CACHE_TTL_FILES = int(os.getenv("CACHE_TTL_FILES", "300"))
CACHE_TTL_TEXTS = int(os.getenv("CACHE_TTL_TEXTS", "300"))
CACHE_TTL_SUMMARIES = int(os.getenv("CACHE_TTL_SUMMARIES", "3600"))

MIN_CHARS = int(os.getenv("MIN_CHARS", "140"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "200"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "220"))

ALLOW_GENERAL_KNOWLEDGE = os.getenv("ALLOW_GENERAL_KNOWLEDGE", "1") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes (OPENAI_API_KEY, GITHUB_REPO)")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

def log(*args):
    if DEBUG:
        print("[DEBUG]", *args)

def limpar_texto(txt: str) -> str:
    txt = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")
    txt = re.sub(r"\n{2,}", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

def _github_headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

_cache_files = {"ts": 0.0, "items": []}
_cache_texts = {"ts": 0.0, "items": []}          # List[Tuple[url, text]]
_cache_summaries = {"ts": 0.0, "sig": "", "items": []}  # List[str] summaries

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
        if item.get("type") == "file" and item.get("name", "").endswith(".txt") and item.get("download_url"):
            arquivos.append(item["download_url"])
        elif item.get("type") == "dir" and item.get("path"):
            arquivos.extend(listar_txt(item["path"]))

    if path == "":
        _cache_files["ts"] = now
        _cache_files["items"] = list(arquivos)

    return arquivos

def ler_txts() -> List[Tuple[str, str]]:
    now = time.time()
    if _cache_texts["items"] and (now - float(_cache_texts["ts"]) < CACHE_TTL_TEXTS):
        return list(_cache_texts["items"])

    textos: List[Tuple[str, str]] = []
    urls = listar_txt()
    log("txt urls:", len(urls))

    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text:
                textos.append((url, limpar_texto(r.text)))
        except Exception:
            pass

    _cache_texts["ts"] = now
    _cache_texts["items"] = list(textos)
    log("txt carregados:", len(textos))
    return textos

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

def _signature_texts(pairs: List[Tuple[str, str]]) -> str:
    h = hashlib.sha1()
    for url, t in pairs:
        h.update(url.encode("utf-8"))
        h.update(str(len(t)).encode("utf-8"))
        h.update(t[:2000].encode("utf-8", "ignore"))
        h.update(t[-800:].encode("utf-8", "ignore"))
        h.update(b"\n--\n")
    return h.hexdigest()

def construir_resumos() -> Tuple[List[str], str]:
    """
    MAP: gera um resumo curto por arquivo .txt.
    Cacheia por TTL ou quando assinatura muda.
    """
    now = time.time()
    pairs = ler_txts()
    sig = _signature_texts(pairs)

    if (
        _cache_summaries["items"]
        and (now - float(_cache_summaries["ts"]) < CACHE_TTL_SUMMARIES)
        and _cache_summaries["sig"] == sig
    ):
        return list(_cache_summaries["items"]), sig

    summaries: List[str] = []
    system = (
        "Você é um resumidor técnico. "
        "Resuma APENAS o que está no texto, sem inventar nada."
    )

    for url, text in pairs:
        # limita o tamanho por arquivo pra não explodir custo
        sample = text[:6000]
        user = (
            "Resuma o texto em 3 a 5 bullets curtos (frases pequenas), "
            "mantendo somente informações presentes. Não copie trechos longos.\n\n"
            f"TEXTO:\n{sample}"
        )
        s = _call_openai(system, user, max_output_tokens=220)
        s = limpar_texto(s)
        summaries.append(f"ARQUIVO: {url}\n{s}")

    _cache_summaries["ts"] = now
    _cache_summaries["sig"] = sig
    _cache_summaries["items"] = list(summaries)
    log("Resumos gerados:", len(summaries))
    return summaries, sig

def gerar_resposta(tema: str, summaries: List[str]) -> str:
    """
    REDUCE: usa TODOS os resumos para responder.
    """
    base = "\n\n---\n\n".join(summaries)
    # limita base pra caber bem
    base = base[:22000]

    if ALLOW_GENERAL_KNOWLEDGE:
        policy = (
            "Você pode complementar com conhecimento geral quando o tema não aparecer nos resumos. "
            "Quando fizer isso, comece com 'Em geral,' e NÃO diga que veio dos arquivos. "
            "Não contradiga os resumos."
        )
    else:
        policy = "Use SOMENTE o que aparece nos resumos."

    system = (
        "Você é um redator técnico. Escreva em português correto, natural, sem soar robótico. "
        + policy
    )

    user = (
        f"TEMA: {tema}\n\n"
        "RESUMOS DA BASE (cobrem todos os .txt):\n"
        f"{base}\n\n"
        "TAREFA:\n"
        f"- Gere 1 único parágrafo com {MIN_CHARS} a {MAX_CHARS} caracteres (com espaços).\n"
        "- Se o tema não estiver na base, diga 'Em geral,' e explique de forma curta.\n"
        "- Se o tema estiver, priorize a base e reformule.\n"
        "- Termine com ponto final.\n"
    )

    texto = _call_openai(system, user, max_output_tokens=MAX_OUTPUT_TOKENS)
    texto = limpar_texto(texto)
    if texto and texto[-1] not in ".!?":
        texto = texto.rstrip(" ,;:-") + "."
    return texto

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json(silent=True) or {}
    tema = (data.get("input", "") or "").strip()

    if not tema:
        return app.response_class(
            response=json.dumps({"error": "tema_vazio"}, ensure_ascii=False),
            mimetype="application/json"
        )

    try:
        summaries, sig = construir_resumos()
        if not summaries:
            return app.response_class(
                response=json.dumps({"reply": "Não encontrei arquivos .txt válidos na base."}, ensure_ascii=False),
                mimetype="application/json"
            )

        texto = gerar_resposta(tema, summaries)

        return app.response_class(
            response=json.dumps({"reply": texto}, ensure_ascii=False),
            mimetype="application/json"
        )

    except Exception as e:
        traceback.print_exc()
        return app.response_class(
            response=json.dumps({"error": "server_failed", "detail": str(e)}, ensure_ascii=False),
            mimetype="application/json"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
