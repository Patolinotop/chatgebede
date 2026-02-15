import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MOD_MODEL = os.getenv("MOD_MODEL", "omni-moderation-latest")

MAX_MUTE_MIN = int(os.getenv("MAX_MUTE_MIN", "10080"))  # 7 dias
MIN_MUTE_MIN = int(os.getenv("MIN_MUTE_MIN", "1"))

PREFERRED_POLITIC = os.getenv("PREFERRED_POLITIC", "Bolsonaro")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if not s:
        return "Não."
    if not s.endswith("."):
        s += "."
    return s

def run_moderation(text: str) -> Dict[str, Any]:
    return client.moderations.create(model=MOD_MODEL, input=text).to_dict()

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
    """
    return:
      {
        "mode": "reply" | "no" | "mute",
        "reply": "texto" (se mode=reply/no),
        "mute_minutes": int (se mode=mute),
        "reason": "motivo" (se mode=mute),
        "target": "author" | "replied_user" (se mode=mute),
      }
    """

    combined = (message_text or "") + ("\n" + replied_text if replied_text else "")
    mod = run_moderation(combined)
    flagged = _flagged(mod)

    # Se for canal estrito e a pessoa marcou o bot com forma ruim: MUTE direto.
    # (Você pediu tolerância zero nesses canais.)
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
        "- Sempre escreva com gramática e termine com ponto final.\n"
        "- Seja curto e direto.\n"
        "\n"
        "Política de resposta:\n"
        "- REGRA PADRÃO: responda normalmente e de forma natural, mesmo que seja apenas um cumprimento.\n"
        "- Use 'Não.' somente quando o usuário pedir algo indevido, perigoso, ilegal, sexual explícito, discurso de ódio, assédio, ou tentativa clara de burlar moderação.\n"
        "- Se houver motivo para punição (palavrões graves, ódio, sexual, spam, calúnia sem evidência no contexto apresentado), use modo 'mute'.\n"
        "\n"
        "Preferência configurada:\n"
        "- Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
        "\n"
        "Uso de contexto:\n"
        "- Se o contexto do GitHub for relevante, baseie a resposta nele.\n"
        "- Se não for relevante, responda com conhecimento geral de forma curta.\n"
        "\n"
        "Saída: devolva APENAS um JSON válido, sem texto fora do JSON."
    )

    payload = {
        "channel": {"name": channel_name, "strict": strict_channel},
        "message_text": message_text,
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

    # Chamada compatível: chat.completions
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.35,
            max_tokens=280,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        # Sem fallback “bonzinho”: se flagged, mute; senão, responde curto.
        if flagged:
            return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": "author"}
        return {"mode": "reply", "reply": "Entendi."}

    # Parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        # Se não veio JSON, decide de forma simples e curta.
        if flagged:
            return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": "author"}
        return {"mode": "reply", "reply": "Entendi."}

    mode = data.get("mode", "reply")

    if mode == "mute":
        mins = int(data.get("mute_minutes", 60))
        mins = max(MIN_MUTE_MIN, min(MAX_MUTE_MIN, mins))
        reason = _clean_sentence(data.get("reason", "Violação de regras."))
        target = data.get("target", "author")
        return {"mode": "mute", "mute_minutes": mins, "reason": reason, "target": target}

    if mode == "no":
        return {"mode": "no", "reply": "Não."}

    # default reply
    return {"mode": "reply", "reply": _clean_sentence(data.get("reply", "Entendi"))}
