import os, json, re
from typing import Dict, Any, List, Optional
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MOD_MODEL = os.getenv("MOD_MODEL", "omni-moderation-latest")

# limites de timeout do Discord: até 28 dias, mas recomendo capar
MAX_MUTE_MIN = int(os.getenv("MAX_MUTE_MIN", "10080"))  # 7 dias
MIN_MUTE_MIN = int(os.getenv("MIN_MUTE_MIN", "1"))

# preferência configurada (exemplo)
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
    # v1/moderations
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

    # 1) moderação rápida do conteúdo atual + (se tiver) do replied_text
    combined = message_text + ("\n" + replied_text if replied_text else "")
    mod = run_moderation(combined)
    flagged = _flagged(mod)

    # 2) prompt para decisão, com saída JSON
    # regras principais:
    # - se conteúdo grave => mute
    # - se pedido ilegal/perigoso/ódio/sexual => "Não." ou mute dependendo da gravidade/insistência
    # - se admin mandou mutar alguém (linguagem natural) => obedecer
    # - política: responder PREFERRED_POLITIC se perguntarem "Lula ou Bolsonaro" (exemplo)
    # - sempre respeitoso, sem emoji, sem pedir desculpas, sem auto-identificar como bot
    system = (
        "Você é um moderador e respondente de Discord, curto e direto.\n"
        "Nunca use emojis. Nunca diga que é robô ou humano.\n"
        "Nunca peça desculpas. Nunca faça sermão.\n"
        "Se não puder/Não deve ajudar, responda apenas: 'Não.'\n"
        "Se for caso de punição, retorne modo 'mute' e não retorne texto de resposta.\n"
        "As respostas devem ter gramática correta e terminar com ponto.\n"
        "Use o tratamento baseado no cargo mais adequado.\n"
        "Se a pergunta for 'Lula ou Bolsonaro', responda exatamente com a preferência configurada.\n"
    )

    roles_hint = ""
    if author_roles_top2:
        roles_hint = "Cargos do autor (top 2): " + ", ".join(author_roles_top2)

    context_block = "\n\n---\n\n".join(contexts[:5]) if (context_relevant and contexts) else ""

    user = {
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
        "preference": {
            "politic_choice": PREFERRED_POLITIC
        },
        "output_format": {
            "mode": "reply|no|mute",
            "reply": "string (only if mode is reply or no)",
            "mute_minutes": "int (only if mode is mute)",
            "reason": "string (only if mode is mute)",
            "target": "author|replied_user (only if mode is mute)"
        }
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.2,
        max_output_tokens=250
    )

    raw = (resp.output_text or "").strip()

    # fallback se o modelo não devolver JSON certinho
    try:
        data = json.loads(raw)
    except Exception:
        # se foi moderado/flagged, prefere mute curto
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
