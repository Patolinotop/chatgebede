import os
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROFANITY_RE = re.compile(
    r"\b("
    r"fdp|f\.?d\.?p|"
    r"puta|put@|"
    r"caralho|krl|kr(l+)?|"
    r"porra|"
    r"buceta|"
    r"arrombado|"
    r"filho da puta|"
    r"desgra(ç|c)a|"
    r"vai tomar no cu|vt(nc|mnc)|"
    r"\bcu\b|"
    r"merda"
    r")\b",
    flags=re.IGNORECASE
)

HATE_HINT_RE = re.compile(
    r"\b("
    r"naz(i|ista)|hitler|"
    r"macaco|preto imundo|"
    r"viad(o|a)|"
    r"matar (grupo)|exterminar (grupo)"
    r")\b",
    flags=re.IGNORECASE
)

SEXUAL_HINT_RE = re.compile(
    r"\b("
    r"sexo|transar|nude|nudes|"
    r"pinto|pau|piroca|"
    r"buceta|"
    r"gozar|"
    r"punheta|"
    r"estupro"
    r")\b",
    flags=re.IGNORECASE
)

BANNED_GENERIC_REPLIES = {
    "entendido.",
    "confirmado.",
    "ignorado.",
    "mensagem sem infração e sem necessidade de resposta.",
    "mensagem sem infração.",
    "sem infração.",
    "sem necessidade de resposta.",
    "não configura infração.",
    "não apresenta infração.",
}

BANNED_GENERIC_PATTERNS = [
    r"mensagem sem infração",
    r"sem necessidade de resposta",
    r"não configura infração",
    r"não apresenta infração",
]

def _clean(s: Optional[str]) -> str:
    return (s or "").replace("\u200b", "").strip()

def _json_load_safe(txt: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(txt)
    except Exception:
        return None

def _has_banned_generic(reply: str) -> bool:
    low = (reply or "").strip().lower()
    if low in BANNED_GENERIC_REPLIES:
        return True
    for pat in BANNED_GENERIC_PATTERNS:
        if re.search(pat, low, flags=re.IGNORECASE):
            return True
    return False

def _make_style_rule(strict_channel: bool, style_flags: Dict[str, bool]) -> Optional[str]:
    if not strict_channel:
        return None
    reasons = []
    if style_flags.get("emoji"):
        reasons.append("Uso de emojis em canal formal.")
    if style_flags.get("slang"):
        reasons.append("Uso de gírias em canal formal.")
    if style_flags.get("caps"):
        reasons.append("Uso de caixa alta excessiva em canal formal.")
    if not style_flags.get("punctuation"):
        reasons.append("Falta de pontuação básica em canal formal.")
    return " ".join(reasons) if reasons else None

def _rewrite_reply_via_model(original: str, user_payload: Dict[str, Any]) -> str:
    """
    Segunda chamada: reescrever para algo coerente, curto e direto,
    SEM usar frases genéricas proibidas e SEM puxar assunto.
    """
    sys = """
Reescreva a resposta para ser curta, fria, direta e coerente com a conversa.
Regras:
- Não use emojis.
- Não diga que é robô/IA.
- Não use frases genéricas como: "Entendido", "Confirmado", "Ignorado",
  "Mensagem sem infração", "Sem necessidade de resposta", "Não configura infração".
- Não puxe assunto e não faça perguntas, a menos que o usuário tenha feito pergunta explícita.
- Sempre responda algo útil ao conteúdo.
Responda apenas com o texto final, sem aspas.
""".strip()

    usr = {
        "bad_reply": original,
        "conversation": user_payload,
        "instruction": "Reescreva de forma coerente seguindo as regras."
    }

    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(usr, ensure_ascii=False)},
        ],
        temperature=0.25,
        max_tokens=120,
    )
    out = _clean(r.choices[0].message.content)
    return out

def decide_action(
    message_text: str,
    author_display: str,
    author_roles_top2: List[str],
    replied_user_display: Optional[str],
    replied_text: str,
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

    msg = _clean(message_text)
    rep = _clean(replied_text)

    style_rule = _make_style_rule(strict_channel, style_flags)

    profanity_here = bool(PROFANITY_RE.search(msg))
    profanity_replied = bool(PROFANITY_RE.search(rep)) if rep else False

    hate_here = bool(HATE_HINT_RE.search(msg))
    hate_replied = bool(HATE_HINT_RE.search(rep)) if rep else False

    sexual_here = bool(SEXUAL_HINT_RE.search(msg))
    sexual_replied = bool(SEXUAL_HINT_RE.search(rep)) if rep else False

    spam_repeat = bool(spam_flags.get("repeat_spam"))
    emoji_flood = bool(spam_flags.get("emoji_flood_single"))

    rag_block = ""
    if contexts:
        rag_block = "\n\n".join(contexts[:5])

    system = f"""
Você é um moderador e assistente de servidor Discord.

Você sempre escolhe UMA ação: reply, mute.
O modo "no" NÃO deve ser usado. Se você iria dizer "Não.", use mode="reply" com uma frase seca.

Regras de personalidade:
- Curto, frio, direto e sempre com gramática.
- Sem emojis.
- Não diga que é robô/IA.
- Proibido responder com: "Entendido", "Confirmado", "Ignorado",
  "Mensagem sem infração", "Sem necessidade de resposta", "Não configura infração".
- Não puxe assunto e não faça perguntas, a menos que o usuário tenha feito pergunta explícita.
- Toda mensagem marcada para você tem necessidade de resposta (exceto quando a ação é mute).

Regras de moderação:
- Se houver infração no replied_text (mensagem respondida), e o usuário marcou você como denúncia,
  você DEVE considerar punir o autor da mensagem respondida (target="replied_user") mesmo que o texto do denunciante seja normal.
- Se houver infração no texto atual de quem marcou, o alvo é target="author".
- Infrações graves: ódio, sexual explícito, ameaças, calúnia sem evidência, assédio pesado, palavrões graves.
- Spam: 3 mensagens repetidas no canal em pouco tempo OU flood de 7+ emojis numa mensagem.
- Canal formal: se strict_channel=True e houver violações de estilo ao te marcar, isso pode ser infração.
- Se is_admin_author=True, obedeça pedidos diretos do admin para moderar quando fizer sentido.

Saída: APENAS JSON válido no schema abaixo:
{{
  "mode": "reply" | "mute",
  "reply": "texto" (obrigatório se mode="reply"),
  "mute_minutes": inteiro (obrigatório se mode="mute"),
  "reason": "motivo curto" (obrigatório se mode="mute"),
  "target": "author" | "replied_user" (obrigatório se mode="mute")
}}

Restrições do reply:
- Não mencione {author_display} a menos que seja necessário.
- Se mencionar alguém, use author_mention ou replied_user_mention.
- Não coloque ponto extra em cima de "!" ou "?".
""".strip()

    user_payload = {
        "author_display": author_display,
        "author_mention": author_mention,
        "author_roles_top2": author_roles_top2,
        "channel_name": channel_name,
        "strict_channel": strict_channel,
        "implicit_reply": implicit_reply,

        "message_text": msg,
        "replied_user_display": replied_user_display,
        "replied_user_mention": replied_user_mention,
        "replied_text": rep,

        "last5_author": last5_author[-5:],
        "last5_other": last5_other[-5:],

        "is_admin_author": is_admin_author,

        "rag_relevant": context_relevant,
        "rag_context": rag_block,

        "signals": {
            "style_rule": style_rule,
            "profanity_here": profanity_here,
            "profanity_replied": profanity_replied,
            "hate_here": hate_here,
            "hate_replied": hate_replied,
            "sexual_here": sexual_here,
            "sexual_replied": sexual_replied,
            "spam_repeat": spam_repeat,
            "emoji_flood": emoji_flood,
        },

        "instruction": """
Decida a ação:

1) Avalie infração na mensagem respondida (replied_text). Se for infração, mute replied_user.
2) Senão, avalie infração na mensagem atual (message_text). Se for infração, mute author.
3) Se não houver infração, responda SEMPRE em mode="reply":
   - Se for pergunta: responda com explicação clara.
   - Se for conversa/afirmação/reação: responda curto e relevante, sem frases genéricas.
"""
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.3,
        max_tokens=520,
    )

    txt = _clean(resp.choices[0].message.content)
    data = _json_load_safe(txt)

    if not isinstance(data, dict) or "mode" not in data:
        # último recurso: ainda responde, mas seco.
        return {"mode": "reply", "reply": "Certo."}

    mode = data.get("mode", "reply")

    if mode == "mute":
        minutes = int(data.get("mute_minutes", 30))
        minutes = max(1, min(minutes, 1440))
        target = data.get("target", "author")
        if target not in ("author", "replied_user"):
            target = "author"
        reason = _clean(data.get("reason", "Violação de regras."))
        return {"mode": "mute", "mute_minutes": minutes, "reason": reason[:220], "target": target}

    # Força reply sempre
    reply = _clean(data.get("reply", "Certo."))
    reply = re.sub(rf"<@!?{bot_user_id}>", "", reply).strip()

    # Remove respostas genéricas proibidas e reescreve via modelo
    if _has_banned_generic(reply):
        rewritten = _rewrite_reply_via_model(reply, user_payload)
        if rewritten:
            reply = rewritten

    # Ajuste final de pontuação sem estragar "!" ou "?"
    if reply.endswith(".."):
        while reply.endswith(".."):
            reply = reply[:-1]
    if reply and reply[-1] not in ".!?":
        reply += "."

    # Se ainda cair em frase proibida, reescreve de novo (uma vez)
    if _has_banned_generic(reply):
        rewritten = _rewrite_reply_via_model(reply, user_payload)
        if rewritten:
            reply = rewritten
            if reply and reply[-1] not in ".!?":
                reply += "."

    return {"mode": "reply", "reply": reply}
