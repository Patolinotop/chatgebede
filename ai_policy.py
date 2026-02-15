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

META_BANNED_PATTERNS = [
    r"mensagem respondida",
    r"não apresenta infração",
    r"sem infração",
    r"não há infração",
    r"apenas cumprimentando",
    r"analis(e|ei|ando) (a|o) mensagem",
    r"denúncia",
    r"report(e|ei|ando)",
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
        return "Entendi."
    s = re.sub(r"([!?\.])\.+$", r"\1", s)
    if re.search(r"[.!?]$", s):
        return s
    if re.search(r"[,;:]$", s):
        return re.sub(r"[,;:]+$", ".", s)
    return s + "."

def _strip_meta(reply: str) -> str:
    r = (reply or "").strip()
    low = r.lower()
    if any(re.search(p, low) for p in META_BANNED_PATTERNS):
        return "Entendi."
    return r

def _strip_questions(reply: str) -> str:
    # Remove perguntas (você pediu sem puxar assunto)
    r = (reply or "").strip()
    # se a IA tentar emendar pergunta, corta no primeiro '?'
    if "?" in r:
        r = r.split("?", 1)[0].strip()
        if not r:
            r = "Entendi."
        # garante ponto
        if not re.search(r"[.!?]$", r):
            r += "."
    return r

def _enforce_brevity(reply: str) -> str:
    """
    Regra:
    - padrão: 1 frase curta
    - se tiver muito longo, corta para ~14 palavras
    - nunca termina com ':' ';' ','
    """
    r = (reply or "").strip()
    # remove quebras exageradas
    r = re.sub(r"\s+", " ", r).strip()

    # se tiver múltiplas frases, mantém no máximo 2
    parts = re.split(r"(?<=[.!])\s+", r)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 2:
        r = " ".join(parts[:2])
    else:
        r = " ".join(parts)

    words = r.split()
    if len(words) > 14:
        r = " ".join(words[:14]).strip()
        if not re.search(r"[.!?]$", r):
            r += "."

    r = re.sub(r"[,;:]+$", ".", r)
    if not re.search(r"[.!?]$", r):
        r += "."
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

    context_block = "\n\n---\n\n".join(contexts[:5]) if (context_relevant and contexts) else ""
    needs_reply = reporter_needs_reply(author_raw_text) if implicit_reply else True

    system = (
        "Você conversa e modera em Discord.\n"
        "Estilo obrigatório:\n"
        "- Nunca use emojis.\n"
        "- Nunca diga que é robô ou humano.\n"
        "- Nunca peça desculpas.\n"
        "- Nunca faça sermão.\n"
        "- Nunca puxe assunto.\n"
        "- Nunca faça perguntas de volta.\n"
        "- Não ofereça ajuda extra no final.\n"
        "- Seja curto e direto.\n"
        "- Regra de tamanho: 1 frase curta por padrão. No máximo 2 frases se necessário.\n"
        "- Gramática correta; termine com '.', '!' ou '?'.\n"
        "\n"
        "PROIBIDO:\n"
        "- Frases meta como 'sem infração', 'mensagem respondida', 'denúncia', 'análise', 'relatório'.\n"
        f"- Mencionar o próprio bot (<@{bot_user_id}> ou <@!{bot_user_id}>).\n"
        "- Vocativos com nomes (ex.: 'Editi,').\n"
        "\n"
        "Decisão:\n"
        "- Se houver infração: mode='mute'.\n"
        "- Se NÃO houver infração: mode='reply' e responda sobre o conteúdo, sem meta.\n"
        "- Use 'Não.' apenas para pedidos indevidos (ilegal/perigoso/sexual explícito/ódio/assédio/burlar regras).\n"
        "\n"
        "Reply implícito:\n"
        "- Se implicit_reply=true, trate como conversa sobre a mensagem respondida.\n"
        "- Se a infração estiver na mensagem respondida, aplique mute no autor dela (target='replied_user').\n"
        "- Se não houver infração:\n"
        "  - Se reporter_needs_reply=false, responda apenas ao replied_user (sem mencionar o autor do reply).\n"
        "  - Se reporter_needs_reply=true, pode responder ambos, usando apenas as menções fornecidas.\n"
        "\n"
        "Preferência:\n"
        "- Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
        "\n"
        "Saída: devolva APENAS JSON válido."
    )

    payload = {
        "channel": {"name": channel_name, "strict": strict_channel},
        "message_text": message_text,
        "author_raw_text": author_raw_text,
        "implicit_reply": implicit_reply,
        "reporter_needs_reply": needs_reply,
        "author_mention": author_mention,
        "replied_user_mention": replied_user_mention,
        "replied_text": replied_text or "",
        "last5_author": last5_author,
        "last5_other": last5_other,
        "context_relevant": context_relevant,
        "context": context_block,
        "flagged_by_moderation": flagged,
        "preference": {"politic_choice": PREFERRED_POLITIC},
        "output_format": {
            "mode": "reply|no|mute",
            "reply": "string",
            "mute_minutes": "int",
            "reason": "string",
            "target": "author|replied_user"
        }
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.25,
            max_tokens=220,
            timeout=OPENAI_REQ_TIMEOUT
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        if flagged:
            return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": "author"}
        return {"mode": "reply", "reply": "Entendi."}

    try:
        data = json.loads(raw)
    except Exception:
        if flagged:
            return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": "author"}
        return {"mode": "reply", "reply": "Entendi."}

    mode = data.get("mode", "reply")

    if mode == "mute":
        mins = int(data.get("mute_minutes", 60))
        mins = max(MIN_MUTE_MIN, min(MAX_MUTE_MIN, mins))
        reason = _clean_sentence(data.get("reason", "Violação de regras"))
        target = data.get("target", "author")
        return {"mode": "mute", "mute_minutes": mins, "reason": reason, "target": target}

    if mode == "no" and implicit_reply:
        mode = "reply"

    if mode == "no":
        return {"mode": "no", "reply": "Não."}

    reply = _clean_sentence(data.get("reply", "Entendi."))
    reply = _strip_meta(reply)
    reply = _strip_questions(reply)      # ✅ sem puxar assunto
    reply = _enforce_brevity(reply)      # ✅ curto por padrão
    return {"mode": "reply", "reply": reply}
