# ================================
# MENU EB – Backend Chatbot API (ANÁLISE SEMÂNTICA CONTROLADA)
# STATUS: ATUALIZADO – MENOS ALUCINAÇÃO E SEM "CORTE DO NADA"
#
# O QUE FOI ARRUMADO (SEM AUMENTAR LIMITE DE GERAÇÃO):
# - O texto NÃO é mais cortado na marra por caracteres
# - Se vier longo/curto/incompleto, faz 2ª chamada pedindo REESCRITA 120–160 chars
# - Detector de resposta "cortada" (termina com vírgula, conjunção, preposição, etc.)
# - Contexto por relevância para reduzir mistura de assuntos
# - Modelo padrão melhor custo/benefício: gpt-4o-mini (troca via OPENAI_MODEL)
# ================================

from flask import Flask, request
import os, json, re, requests, traceback, time
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# ================================
# ENV
# ================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # opcional

CACHE_TTL_FILES = int(os.getenv("CACHE_TTL_FILES", "300"))  # 5 min
CACHE_TTL_TEXTS = int(os.getenv("CACHE_TTL_TEXTS", "300"))  # 5 min

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
    for termo in TERMOS_FIXOS:
        pattern = r"(?<!\w)" + re.escape(termo) + r"(?!\w)"
        texto = re.sub(pattern, termo, texto, flags=re.IGNORECASE)
    return texto

def normalizar_palavras(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-zà-ú0-9\s]", " ", s, flags=re.IGNORECASE)
    parts = [p for p in s.split() if len(p) >= 3]
    stop = {
        "para", "com", "sem", "sobre", "como", "qual", "quais", "porque", "porquê",
        "uma", "uns", "umas", "dos", "das", "que", "não", "nos", "nas", "entre",
        "sua", "seu", "suas", "seus", "esse", "essa", "isso", "aquele", "aquela"
    }
    return [p for p in parts if p not in stop]

def pontuar_relevancia(tema: str, trecho: str) -> int:
    t = set(normalizar_palavras(tema))
    if not t:
        return 0
    x = set(normalizar_palavras(trecho))
    return len(t.intersection(x))

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
        # se terminar com vírgula/ dois-pontos/ ponto-e-vírgula etc, substitui por ponto
        texto = re.sub(r"[,:;–—-]\s*$", "", texto).strip()
        if texto and texto[-1] not in ".!?":
            texto += "."
    return texto

def contar_chars(texto: str) -> int:
    return len(texto)

def resposta_parece_cortada(texto: str) -> bool:
    """
    Heurística: detecta finais típicos de truncamento.
    """
    t = texto.strip()
    if not t:
        return True

    # termina com pontuação "aberta" ou vírgula
    if re.search(r"[,;:–—-]\s*$", t):
        return True

    # termina com palavra “pendente”
    ultima = re.sub(r"[^\wà-ú]+$", "", t.lower()).split()[-1] if t.split() else ""
    pendentes = {
        "e", "ou", "para", "por", "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
        "em", "ao", "aos", "à", "às", "com", "sem", "sobre", "entre", "que"
    }
    if ultima in pendentes:
        return True

    # se não termina com .!? e tem cara de frase longa, pode ter truncado
    if t[-1] not in ".!?" and len(t) >= 90:
        return True

    return False

def dentro_da_faixa(texto: str, min_c: int = 120, max_c: int = 160) -> bool:
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
    textos = ler_txts()
    trechos: List[Tuple[int, str]] = []

    for t in textos:
        for ch in dividir_em_trechos(t, chunk_size=900):
            score = pontuar_relevancia(tema, ch)
            if score > 0:
                trechos.append((score, ch))

    if not trechos and textos:
        # fallback: poucos trechos iniciais (sem misturar tudo)
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
# OpenAI – RACIOCÍNIO CONTROLADO (sem corte brusco)
# ================================
def _call_openai(system_prompt: str, user_prompt: str, max_output_tokens: int = 120) -> str:
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
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
        "Se um detalhe não estiver explícito, OMITE esse detalhe. "
        "Você deve entregar texto curto e finalizado, sem cortes."
    )

    user_prompt = (
        f"TEMA/PERGUNTA: {tema}\n\n"
        "DOCUMENTOS (TRECHOS RELEVANTES):\n"
        f"{contexto}\n\n"
        "REGRAS DE RESPOSTA (OBRIGATÓRIO):\n"
        "1) Responda APENAS sobre o tema.\n"
        "2) Use SOMENTE o que está nos documentos (sem inferir).\n"
        "3) Não copie frases inteiras; reescreva.\n"
        "4) Produza 1 único parágrafo com 120 a 160 caracteres (contando espaços).\n"
        "5) Termine com ponto final.\n"
        "6) Se o tema for amplo, foque no conceito central presente nos documentos.\n"
    )

    try:
        texto = _call_openai(system_prompt, user_prompt, max_output_tokens=120)
        if not texto:
            raise RuntimeError("Resposta vazia da OpenAI")

        texto = aplicar_capitalizacao(texto)
        texto = garantir_pontuacao_final(texto)

        # Se não ficou dentro da faixa OU parece truncada -> reescrita guiada (sem aumentar tokens)
        if (not dentro_da_faixa(texto, 120, 160)) or resposta_parece_cortada(texto):
            rewrite_prompt = (
                f"TEMA/PERGUNTA: {tema}\n\n"
                "DOCUMENTOS (TRECHOS RELEVANTES):\n"
                f"{contexto}\n\n"
                "TEXTO ATUAL (NÃO CONFIE NELE SE ESTIVER LONGO/CORTADO):\n"
                f"{texto}\n\n"
                "TAREFA:\n"
                "- REESCREVA o conteúdo acima para ficar ENTRE 120 e 160 caracteres (com espaços).\n"
                "- Mantenha SOMENTE informações explícitas nos documentos.\n"
                "- 1 único parágrafo, formal, objetivo.\n"
                "- Termine com ponto final.\n"
                "- Não use vírgula no final nem deixe frase incompleta.\n"
            )
            texto2 = _call_openai(system_prompt, rewrite_prompt, max_output_tokens=120).strip()
            if texto2:
                texto2 = aplicar_capitalizacao(texto2)
                texto2 = garantir_pontuacao_final(texto2)

                # se a reescrita melhorou, usa ela
                if dentro_da_faixa(texto2, 120, 160) and not resposta_parece_cortada(texto2):
                    texto = texto2
                else:
                    # último ajuste leve: se passou um pouquinho, tenta limpar excesso sem cortar no meio
                    # (ainda assim sem inventar)
                    # remove espaços duplos e garante ponto
                    texto = re.sub(r"\s{2,}", " ", texto2 if texto2 else texto).strip()
                    texto = garantir_pontuacao_final(texto)

        # fallback seguro se ainda ficou ruim
        if (not dentro_da_faixa(texto, 120, 160)) or resposta_parece_cortada(texto):
            texto = "Os documentos disponíveis não especificam, de forma suficiente, detalhes sobre o tema solicitado para afirmar regras adicionais."
            texto = aplicar_capitalizacao(texto)
            texto = garantir_pontuacao_final(texto)

            # garante faixa sem “cortar do nada”: pede reescrita do fallback se necessário
            if not dentro_da_faixa(texto, 120, 160):
                fallback_prompt = (
                    "Reescreva o texto a seguir para ficar ENTRE 120 e 160 caracteres (com espaços), "
                    "1 parágrafo formal e terminando com ponto:\n"
                    f"{texto}"
                )
                texto_fb = _call_openai(
                    "Você é um redator técnico. Mantenha o sentido, sem adicionar fatos novos.",
                    fallback_prompt,
                    max_output_tokens=120
                ).strip()
                if texto_fb:
                    texto_fb = garantir_pontuacao_final(texto_fb)
                    if dentro_da_faixa(texto_fb, 120, 160) and not resposta_parece_cortada(texto_fb):
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
