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

# abreviações comuns de palavrão/mandar se ferrar
ABBR_MAP = {
    r"\bfdp\b": "filho da puta",
    r"\bfds\b": "foda-se",
    r"\bvsf\b": "vai se foder",
    r"\bpqp\b": "puta que pariu",
    r"\bkrl\b": "caralho",
    r"\bcrl\b": "caralho",
    r"\bporra\b": "porra",
    r"\bmerda\b": "merda",
}

def normalize_text(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "")).strip().lower()
    for pat, rep in ABBR_MAP.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if not s:
        return "Não."
    s = re.sub(r"([!?\.])\.+$", r"\1", s)
    if re.search(r"[.!?]$", s):
        return s
    if re.search(r"[,;:]$", s):
        return re.sub(r"[,;:]+$", ".", s)
    return s + "."

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
) -> Dict[str, Any]:

    norm_message = normalize_text(message_text or "")
    norm_replied = normalize_text(replied_text or "") if replied_text else ""
    combined_norm = (norm_message + "\n" + norm_replied).strip()

    mod = run_moderation(combined_norm if combined_norm else (message_text or ""))
    flagged = _flagged(mod)

    # Canal estrito: marcou o bot com forma ruim => mute direto
    if strict_channel and any(bool(v) for v in style_flags.values()):
        return {
            "mode": "mute",
            "mute_minutes": 30,
            "reason": "Forma inadequada para este canal.",
            "target": "author",
        }

    context_block = "\n\n---\n\n".join(contexts[:5]) if (context_relevant and contexts) else ""

    system = (
        "Você conversa e modera em Discord.\n"
        "Estilo obrigatório:\n"
        "- Nunca use emojis.\n"
        "- Nunca diga que é robô ou humano.\n"
        "- Nunca peça desculpas.\n"
        "- Nunca faça sermão.\n"
        "- Nunca escreva frases de atendimento do tipo 'como posso ajudar' ou 'se quiser posso'.\n"
        "- Não ofereça ajuda extra no final. Responda e pare.\n"
        "- Seja curto e direto.\n"
        "- Gramática correta; termine com '.', '!' ou '?'.\n"
        "\n"
        "Interpretação:\n"
        "- Trate abreviações como 'fdp', 'fds', 'vsf', 'pqp', 'krl' como palavrões.\n"
        "\n"
        "Política:\n"
        "- REGRA PADRÃO: responda normalmente, inclusive cumprimentos.\n"
        "- Use 'Não.' somente para pedidos indevidos (ilegal/perigoso/sexual explícito/ódio/assédio/burlar regras).\n"
        "- Se for caso de punição (palavrões graves, ódio, sexual, spam, calúnia sem evidência no contexto apresentado), use modo 'mute'.\n"
        "\n"
        "Preferência:\n"
        "- Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
        "\n"
        "Saída: devolva APENAS JSON válido, sem texto fora do JSON."
    )

    payload = {
        "channel": {"name": channel_name, "strict": strict_channel},
        "message_text": message_text,
        "message_text_normalized": combined_norm,
        "author_display": author_display,
        "author_roles_top2": author_roles_top2,
        "replied_user_display": replied_user_display or "",
        "replied_text": replied_text or "",
        "last5_author": last5_author,
        "last5_other": last5_other,
        "is_admin_author": is_admin_author,
        "admin_targets": admin_targets,
        "context_relevant": context_relevant,
        "context": context_block,
        "style_flags": style_flags,
        "flagged_by_moderation": flagged,
        "preference": {"politic_choice": PREFERRED_POLITIC},
        "output_format": {
            "mode": "reply|no|mute",
            "reply": "string (only if mode is reply or no)",
            "mute_minutes": "int (only if mode is mute)",
            "reason": "string (only if mode is mute)",
            "target": "author|replied_user (only if mode is mute)"
        }
    }

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.35,
            max_tokens=280,
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

    if mode == "no":
        return {"mode": "no", "reply": "Não."}

    return {"mode": "reply", "reply": _clean_sentence(data.get("reply", "Entendi"))}
