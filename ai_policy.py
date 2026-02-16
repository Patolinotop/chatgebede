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

META_PHRASES = [
    r"\bno contexto\b",
    r"\bgeralmente\b",
    r"\bindica\b",
    r"\bé uma afirmação que\b",
    r"\busada para\b",
    r"\binterlocutor\b",
    r"\bempatia\b",
    r"\bdesvalorizar\b",
    r"\bprovocar\b",
    r"\bnão (?:configura|caracteriza) infração\b",
    r"\b(não )?apresenta infração\b",
    r"\bsem infração\b",
    r"\bmensagem respondida\b",
    r"\bdenúncia\b",
    r"\banálise\b",
    r"\brelatório\b",
]

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
        return ""
    s = re.sub(r"([!?\.])\.+$", r"\1", s)
    s = re.sub(r"[,;:]+$", ".", s)
    if not re.search(r"[.!?]$", s):
        s += "."
    return s

def strip_meta_phrases(text: str) -> str:
    t = (text or "")
    for p in META_PHRASES:
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def contains_meta(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in META_PHRASES)

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

def classify_intent(topic_text: str) -> str:
    t = (topic_text or "").strip()
    if not t:
        return "chat"
    low = t.lower()
    if "?" in t:
        return "question"
    starters = ("o que", "oq", "quem", "onde", "quando", "por que", "porque", "como", "qual", "quais")
    if low.startswith(starters):
        return "question"
    if re.search(r"\b(explique|define|defina|diga|fale|mostre|ensine|resuma)\b", low):
        return "question"
    return "chat"

def rewrite_short_no_meta(topic_text: str, draft: str, intent: str) -> str:
    """
    Segunda chamada (só quando necessário):
    reescreve a resposta:
    - intent='chat': 1 frase curta
    - intent='question': tamanho livre, mas direto
    - sem meta/professor
    - sem pergunta de volta
    """
    system = (
        "Reescreva a resposta para Discord.\n"
        "- Nunca use emojis.\n"
        "- Nunca peça desculpas.\n"
        "- Nunca diga que é robô/humano.\n"
        "- Nunca faça perguntas de volta.\n"
        "- Nunca use texto meta (contexto, geralmente, indica, sem infração, análise, relatório).\n"
        "- Seja frio, direto e com gramática.\n"
        "- Se intent='chat': devolva 1 frase curta reagindo ao conteúdo.\n"
        "- Se intent='question': responda o necessário, direto.\n"
        "Saída: somente o texto final, sem JSON."
    )
    payload = {
        "intent": intent,
        "topic_text": topic_text,
        "draft": draft
    }
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_tokens=220 if intent == "chat" else 650,
        timeout=OPENAI_REQ_TIMEOUT
    )
    return (resp.choices[0].message.content or "").strip()

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

    if spam_flags.get("repeat_spam") or spam_flags.get("emoji_flood_single") or spam_flags.get("emoji_flood_window"):
        return {"mode": "mute", "mute_minutes": 15, "reason": "Spam ou flood.", "target": "author"}

    norm_message = normalize_text(message_text or "")
    norm_replied = normalize_text(replied_text or "") if replied_text else ""
    combined_norm = (norm_message + "\n" + norm_replied).strip()

    mod = run_moderation(combined_norm if combined_norm else (message_text or ""))
    flagged = _flagged(mod)

    if strict_channel and any(bool(v) for v in style_flags.values()):
        return {"mode": "mute", "mute_minutes": 30, "reason": "Forma inadequada para este canal.", "target": "author"}

    if flagged:
        target = "replied_user" if (implicit_reply and replied_text) else "author"
        return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": target}

    context_block = "\n\n---\n\n".join(contexts[:5]) if (context_relevant and contexts) else ""
    needs_reply = reporter_needs_reply(author_raw_text) if implicit_reply else True

    topic_text = replied_text.strip() if (implicit_reply and replied_text) else (message_text or "").strip()
    intent = classify_intent(topic_text)

    system = (
        "Você conversa e modera em Discord.\n"
        "\n"
        "Regras fixas:\n"
        "- Nunca use emojis.\n"
        "- Nunca diga que é robô ou humano.\n"
        "- Nunca peça desculpas.\n"
        "- Nunca faça sermão.\n"
        "- Nunca ofereça ajuda extra no final.\n"
        "- Nunca faça perguntas de volta.\n"
        "- Seja frio, direto e com gramática.\n"
        "- Sempre termine com '.', '!' ou '?'.\n"
        "\n"
        "PROIBIDO:\n"
        "- Texto meta/professor (no contexto, geralmente, indica, é uma afirmação que, sem infração, análise, relatório).\n"
        f"- Mencionar o próprio bot (<@{bot_user_id}> ou <@!{bot_user_id}>).\n"
        "- Vocativos com nomes (ex.: 'Editi,').\n"
        "\n"
        "Como responder:\n"
        "- Se intent='chat': responda curto (1 frase), reagindo ao conteúdo, sem puxar assunto.\n"
        "- Se intent='question': responda do tamanho necessário para explicar bem.\n"
        "- Responda ao que foi dito. Não explique o 'significado' da frase.\n"
        "\n"
        "Reply:\n"
        "- Se implicit_reply=true e reporter_needs_reply=false, responda apenas ao replied_user usando replied_user_mention.\n"
        "- Se reporter_needs_reply=true, pode responder ambos usando apenas as menções fornecidas.\n"
        "\n"
        "Preferência:\n"
        "- Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
        "\n"
        "Saída: devolva APENAS JSON válido."
    )

    payload = {
        "intent": intent,
        "channel": {"name": channel_name, "strict": strict_channel},
        "implicit_reply": implicit_reply,
        "reporter_needs_reply": needs_reply,
        "author_mention": author_mention,
        "replied_user_mention": replied_user_mention,
        "topic_text": topic_text,
        "message_text": message_text,
        "replied_text": replied_text or "",
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
            max_tokens=900,
            timeout=OPENAI_REQ_TIMEOUT
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        # sem fallback fixo: se der erro de rede, não inventa frase
        return {"mode": "no", "reply": "Não."}

    try:
        data = json.loads(raw)
    except Exception:
        return {"mode": "no", "reply": "Não."}

    mode = data.get("mode", "reply")
    if mode == "no":
        return {"mode": "no", "reply": "Não."}

    draft = (data.get("reply", "") or "").strip()

    # poda meta (sem trocar por texto enlatado)
    pruned = strip_meta_phrases(draft)
    pruned = _clean_sentence(pruned)

    # se ainda ficou meta/feio/vazio, reescreve com 2ª chamada (sem texto pronto)
    if (not pruned) or contains_meta(draft) or len(pruned) < 3:
        try:
            fixed = rewrite_short_no_meta(topic_text=topic_text, draft=draft, intent=intent)
            fixed = _clean_sentence(strip_meta_phrases(fixed))
            if fixed:
                pruned = fixed
        except Exception:
            # aqui, se falhar a reescrita, não inventa frase: retorna 'Não.'
            return {"mode": "no", "reply": "Não."}

    # reply implícito sem marcar denunciante
    if implicit_reply and replied_user_mention and not needs_reply:
        if pruned and not pruned.startswith(replied_user_mention):
            pruned = f"{replied_user_mention} {pruned}"

    return {"mode": "reply", "reply": pruned}
