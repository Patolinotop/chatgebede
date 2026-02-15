import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MOD_MODEL = os.getenv("MOD_MODEL", "omni-moderation-latest")

MAX_MUTE_MIN = int(os.getenv("MAX_MUTE_MIN", "10080"))
MIN_MUTE_MIN = int(os.getenv("MIN_MUTE_MIN", "1"))

PREFERRED_POLITIC = os.getenv("PREFERRED_POLITIC", "Bolsonaro")
OPENAI_REQ_TIMEOUT = float(os.getenv("OPENAI_REQ_TIMEOUT", "20"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ABBR_MAP = {
    r"\bfdp\b": "filho da puta",
    r"\bfds\b": "foda-se",
    r"\bvsf\b": "vai se foder",
    r"\bpqp\b": "puta que pariu",
    r"\bkrl\b": "caralho",
    r"\bcrl\b": "caralho",
}

GENERIC_REPORTER_PATTERNS = [
    r"^\s*$",
    r"^(olha|olha isso|ve|vê|ve ai|vê aí|ve isso|vê isso)\s*$",
    r"^(ta ai|tá aí|ai|aí)\s*$",
    r"^(analisa|analise|avalia|verifica|confere)\s*$",
    r"^(olha ai|olha aí|ve ai|vê aí)\s*$",
    r"^(resolve|da uma olhada|dá uma olhada)\s*$",
]

# Meta “professor” que você NÃO quer
META_HARD_BANNED = [
    r"mensagem respondida",
    r"não apresenta infração",
    r"sem infração",
    r"não há infração",
    r"não configura infração",
    r"configura infração",
    r"denúncia",
    r"relatório",
    r"análise",
    r"analis(e|ei|ando)",
    r"no contexto",
    r"geralmente",
    r"indica",
    r"é uma afirmação que",
    r"usada para",
    r"interlocutor",
    r"empatia",
    r"desvalorizar",
    r"provocar",
]

SHORT_ACKS = ["Entendi.", "Certo.", "Ok.", "Beleza.", "Anotado."]

def normalize_text(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "")).strip().lower()
    for pat, rep in ABBR_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

def reporter_needs_reply(author_raw_text: str) -> bool:
    t = (author_raw_text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    for pat in GENERIC_REPORTER_PATTERNS:
        if re.match(pat, t):
            return False
    return True

def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if not s:
        return "Entendi."
    s = re.sub(r"([!?\.])\.+$", r"\1", s)
    if re.search(r"[.!?]$", s):
        return s
    if re.search(r"[,;:]+$", s):
        return re.sub(r"[,;:]+$", ".", s)
    return s + "."

def _has_meta(reply: str) -> bool:
    low = (reply or "").lower()
    return any(re.search(p, low) for p in META_HARD_BANNED)

def _sanitize_reply(reply: str) -> str:
    r = _clean_sentence(reply)
    if not r or _has_meta(r):
        return "Entendi."
    return r

def run_moderation(text: str) -> Dict[str, Any]:
    return client.moderations.create(
        model=MOD_MODEL,
        input=text,
        timeout=OPENAI_REQ_TIMEOUT
    ).to_dict()

def _flagged(mod: Dict[str, Any]) -> bool:
    try:
        return bool(mod["results"][0]["flagged"])
    except Exception:
        return False

def is_question_like(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    starters = ("o que", "oq", "quem", "onde", "quando", "por que", "porque", "como", "qual", "quais")
    if t.startswith(starters):
        return True
    # pedidos diretos
    if re.search(r"\b(explique|define|defina|diga|fale|mostre|ensine|resuma)\b", t):
        return True
    return False

def decide_action(
    message_text: str,
    author_display: str,
    author_roles_top2: List[str],
    replied_user_display: Optional[str],
    replied_text: Optional[str],
    last5_author: List[str],
    last5_other: List[str],
    is_admin_author: bool,
    admin_targets: List[str],
    contexts: List[str],
    context_relevant: bool,
    channel_name: str,
    strict_channel: bool,
    style_flags: Dict[str, bool],
    spam_flags: Dict[str, Any],
    implicit_reply: bool,
    author_raw_text: str,
    author_mention: str,
    replied_user_mention: str,
    bot_user_id: int,
) -> Dict[str, Any]:

    # spam/flood -> mute direto
    if spam_flags.get("repeat_spam") or spam_flags.get("emoji_flood_single") or spam_flags.get("emoji_flood_window"):
        return {"mode": "mute", "mute_minutes": 15, "reason": "Spam ou flood.", "target": "author"}

    norm_message = normalize_text(message_text or "")
    norm_replied = normalize_text(replied_text or "") if replied_text else ""
    combined_norm = (norm_message + "\n" + norm_replied).strip()

    mod = run_moderation(combined_norm if combined_norm else (message_text or ""))
    flagged = _flagged(mod)

    # canal estrito: forma ruim ao marcar o bot => mute
    if strict_channel and any(bool(v) for v in style_flags.values()):
        return {"mode": "mute", "mute_minutes": 30, "reason": "Forma inadequada para este canal.", "target": "author"}

    # Se moderação marcou como pesado: mute (a IA ainda escolhe duração, mas aqui simplifica)
    if flagged:
        # se foi reply em alguém, pune o replied_user
        target = "replied_user" if (implicit_reply and replied_text) else "author"
        return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": target}

    context_block = "\n\n---\n\n".join(contexts[:5]) if (context_relevant and contexts) else ""
    needs_reply = reporter_needs_reply(author_raw_text) if implicit_reply else True

    # Define qual texto é o “assunto” da conversa
    topic_text = replied_text.strip() if (implicit_reply and replied_text) else (message_text or "").strip()
    questionish = is_question_like(topic_text)

    # Se NÃO é pergunta e NÃO é pedido: resposta curtíssima e acabou (sem IA filosofar)
    if not questionish:
        # ainda permite responder alguém específico no caso de reply
        if implicit_reply and replied_user_mention and not needs_reply:
            return {"mode": "reply", "reply": f"{replied_user_mention} Entendi."}
        return {"mode": "reply", "reply": "Entendi."}

    system = (
        "Você conversa e modera em Discord.\n"
        "Estilo:\n"
        "- Nunca use emojis.\n"
        "- Nunca diga que é robô ou humano.\n"
        "- Nunca peça desculpas.\n"
        "- Nunca faça sermão.\n"
        "- Nunca faça perguntas de volta.\n"
        "- Nunca ofereça ajuda extra no final.\n"
        "- Seja frio, direto e com gramática.\n"
        "- Responda curto por padrão.\n"
        "- Se for uma pergunta que exija explicação, responda detalhado.\n"
        "- Sempre termine com '.', '!' ou '?'.\n"
        "\n"
        "PROIBIDO:\n"
        "- Texto meta (sem infração, contexto, indica, geralmente, análise, relatório).\n"
        f"- Mencionar o próprio bot (<@{bot_user_id}> ou <@!{bot_user_id}>).\n"
        "- Vocativos com nomes (ex.: 'Editi,').\n"
        "\n"
        "Reply:\n"
        "- Se implicit_reply=true, responda sobre o replied_text.\n"
        "- Se reporter_needs_reply=false, responda apenas ao replied_user usando replied_user_mention.\n"
        "- Se reporter_needs_reply=true, pode responder ambos usando as menções fornecidas.\n"
        "\n"
        "Preferência:\n"
        "- Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
        "\n"
        "Saída: devolva APENAS JSON válido."
    )

    payload = {
        "channel": {"name": channel_name, "strict": strict_channel},
        "implicit_reply": implicit_reply,
        "reporter_needs_reply": needs_reply,
        "author_mention": author_mention,
        "replied_user_mention": replied_user_mention,
        "message_text": message_text,
        "replied_text": replied_text or "",
        "topic_text": topic_text,
        "context": context_block,
        "preference": {"politic_choice": PREFERRED_POLITIC},
        "output_format": {"mode": "reply|no", "reply": "string"}
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.25,
            max_tokens=700,
            timeout=OPENAI_REQ_TIMEOUT
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {"mode": "reply", "reply": "Entendi."}

    try:
        data = json.loads(raw)
    except Exception:
        return {"mode": "reply", "reply": "Entendi."}

    mode = data.get("mode", "reply")
    if mode == "no":
        return {"mode": "no", "reply": "Não."}

    reply = _sanitize_reply(data.get("reply", "Entendi."))
    # Se o reply for denúncia genérica e você não quer marcar o autor:
    if implicit_reply and replied_user_mention and not needs_reply:
        # garante que começa marcando o replied_user (mas sem textão meta)
        if not reply.startswith(replied_user_mention):
            reply = f"{replied_user_mention} {reply}"
    return {"mode": "reply", "reply": reply}
