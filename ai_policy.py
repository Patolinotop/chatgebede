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

    # novos campos
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

    # SYSTEM: regras gerais + canal estrito
    system = (
        "Você é moderador e respondente em Discord, curto e direto.\n"
        "Nunca use emojis.\n"
        "Nunca diga que é robô ou humano.\n"
        "Nunca peça desculpas.\n"
        "Nunca faça sermão.\n"
        "Se não puder/não deve ajudar, responda apenas: 'Não.'\n"
        "Se for caso de punição, retorne modo 'mute' e NÃO retorne texto.\n"
        "Respostas devem ter gramática correta e terminar com ponto.\n"
        "Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
        "\n"
        "REGRAS DE CANAL ESTRITO:\n"
        "Se strict_channel=true e houver violação de forma (gíria, emoji, caixa errada, pontuação ruim), aplique mute.\n"
        "Em canal estrito, qualquer menção ao bot com mensagem mal formatada deve resultar em mute.\n"
    )

    roles_hint = ""
    if author_roles_top2:
        roles_hint = "Cargos do autor (top 2): " + ", ".join(author_roles_top2)

    context_block = "\n\n---\n\n".join(contexts[:5]) if (context_relevant and contexts) else ""

    payload = {
        "channel": {"name": channel_name, "strict": strict_channel},
        "message_text": message_text,
        "author_display": author_display,
        "roles_hint": roles_hint,
        "replied_user_display": replied_user_display or "",
        "replied_text": replied_text or "",
        "last5_author": last5_author,
        "last5_other": last5_other,
        "is_admin_author": is_admin_author,
        "admin_targets": admin_targets,
        "context_relevant": context_relevant,
        "context": context_block,
        "style_flags": style_flags,
        "preference": {"politic_choice": PREFERRED_POLITIC},
        "flagged_by_moderation": flagged,
        "output_format": {
            "mode": "reply|no|mute",
            "reply": "string (only if mode is reply or no)",
            "mute_minutes": "int (only if mode is mute)",
            "reason": "string (only if mode is mute)",
            "target": "author|replied_user (only if mode is mute)"
        }
    }

    # Usa CHAT completions (compatível)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=260,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        # se a chamada falhar, a gente retorna erro “seco” ou mute se flagged
        if flagged:
            return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": "author"}
        return {"mode": "no", "reply": "Não."}

    # Parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        # Se canal estrito e style_flags indica problema, muta
        if strict_channel and any(style_flags.values()):
            return {"mode": "mute", "mute_minutes": 30, "reason": "Forma inadequada para este canal.", "target": "author"}
        if flagged:
            return {"mode": "mute", "mute_minutes": 60, "reason": "Conteúdo inadequado.", "target": "author"}
        return {"mode": "no", "reply": "Não."}

    mode = data.get("mode", "no")

    if mode == "mute":
        mins = int(data.get("mute_minutes", 60))
        mins = max(MIN_MUTE_MIN, min(MAX_MUTE_MIN, mins))
        reason = _clean_sentence(data.get("reason", "Violação de regras."))
        target = data.get("target", "author")
        return {"mode": "mute", "mute_minutes": mins, "reason": reason, "target": target}

    if mode == "reply":
        return {"mode": "reply", "reply": _clean_sentence(data.get("reply", ""))}

    return {"mode": "no", "reply": "Não."}
