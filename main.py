# ================================
# MENU EB – Backend Chatbot API (ANÁLISE SEMÂNTICA CONTROLADA)
# STATUS: FINAL – ENTENDE O TEMA, NÃO FOGE, NÃO INVENTA
#
# OBJETIVO REAL (AJUSTADO AO QUE VOCÊ EXPLICOU):
# ✔ Os .txt são a BASE DE CONHECIMENTO
# ✔ O modelo PODE redigir texto novo (não é ctrl+c ctrl+v)
# ✔ MAS só usando informações que EXISTEM nos .txt
# ✔ Pode combinar regras, desde que NÃO se contradigam
# ✔ NÃO responder fora do tema perguntado
# ✔ NÃO dizer "tema não descrito" se o termo existir no texto
# ✔ Se o tema for vago, focar no CONCEITO CENTRAL
# ================================

from flask import Flask, request
import os, json, re, requests, traceback
from dotenv import load_dotenv
from openai import OpenAI

# ================================
# ENV
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

if not OPENAI_API_KEY or not GITHUB_REPO:
    raise RuntimeError("Variáveis de ambiente ausentes")

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
        texto = re.sub(rf"\b{termo.lower()}\b", termo, texto, flags=re.IGNORECASE)
    return texto

# ================================
# GitHub – leitura dos .txt
# ================================

def listar_txt(path=""):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        print("[DEBUG] GitHub API status", r.status_code)
        return []
    arquivos = []
    for item in r.json():
        if item.get("type") == "file" and item.get("name", "").endswith(".txt"):
            arquivos.append(item.get("download_url"))
        elif item.get("type") == "dir":
            arquivos.extend(listar_txt(item.get("path")))
    return arquivos


def ler_contexto():
    textos = []
    for url in listar_txt():
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                textos.append(limpar_texto(r.text))
        except Exception:
            pass
    contexto = " \n".join(textos)
    print("[DEBUG] Contexto size", len(contexto))
    return contexto[:3500]

# ================================
# OpenAI – RACIOCÍNIO CONTROLADO
# ================================

def gerar_resposta(tema: str, contexto: str):
    system_prompt = (
        "Você é um analista de normas institucionais. "
        "Os documentos fornecidos formam a BASE DE CONHECIMENTO. "
        "Você deve COMPREENDER o tema perguntado e redigir um texto coerente, "
        "utilizando apenas informações presentes nesses documentos. "
        "Você pode reorganizar e combinar informações relacionadas, "
        "desde que não crie regras novas nem contradições."
    )

    user_prompt = (
        f"PERGUNTA DO USUÁRIO: {tema}\n\n"
        "DOCUMENTOS DISPONÍVEIS:\n"
        f"{contexto}\n\n"
        "INSTRUÇÕES:\n"
        "- Responda APENAS sobre o tema perguntado\n"
        "- Se o tema for amplo (ex: importância), foque nas funções e regras descritas\n"
        "- NÃO responda com informações não relacionadas ao tema\n"
        "- NÃO copie trechos literalmente\n"
        "- Gere um único parágrafo entre 120 e 160 caracteres\n"
        "- Linguagem formal, humana e objetiva"
    )

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_output_tokens=200
        )

        texto = response.output_text
        if not texto:
            raise RuntimeError("Resposta vazia da OpenAI")

        texto = aplicar_capitalizacao(texto.strip())

        if texto[-1] not in ".!?":
            texto = texto.rsplit(" ", 1)[0] + "."

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
    tema = data.get("input", "").strip()

    if not tema:
        return app.response_class(
            response=json.dumps({"error": "tema_vazio"}, ensure_ascii=False),
            mimetype="application/json"
        )

    contexto = ler_contexto()
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
