import os
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Modelo: mais inteligente que o "mini" baratão, ainda custo ok.
# Você pode sobrescrever no Render com OPENAI_MODEL.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Heurísticas leves (só pra sinalizar; decisão final é do modelo)
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
    r"cu\b|"
    r"merda"
    r")\b",
    flags=re.IGNORECASE
)

HATE_HINT_RE = re.compile(
    r"\b("
    r"naz(i|ista)|hitler|"
    r"macaco|preto imundo|"
    r"viad(o|a)|"
    r"travesti (de forma pejorativa)|"
    r"judeu (de forma pejorativa)|"
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

def _clean(s: Optional[str]) -> str:
    return (s or "").replace("\u200b", "").strip()

def _json_load_safe(txt: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(txt)
    except Exception:
        return None

def _make_style_rule(strict_channel: bool, style_flags: Dict[str, bool]) -> Optional[str]:
    """
    Regra de canal "sério": se o cara mencionou o bot e escreveu no canal estrito,
    gíria/emoji/caixa alta/pontuação ruim pode virar infração.
    Aqui só retornamos um 'sinal' textual; o modelo decide o que fazer.
    """
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

    if reasons:
        return " ".join(reasons)
    return None

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
    """
    Retorna dict:
      - {"mode":"reply","reply":"..."}
      - {"mode":"no","reply":"Não."}
      - {"mode":"mute","mute_minutes":30,"reason":"...","target":"author"|"replied_user"}
    """

    msg = _clean(message_text)
    rep = _clean(replied_text)

    # Sinais "não-modelo" (só dica pro modelo)
    style_rule = _make_style_rule(strict_channel, style_flags)

    profanity_here = bool(PROFANITY_RE.search(msg))
    profanity_replied = bool(PROFANITY_RE.search(rep)) if rep else False

    hate_here = bool(HATE_HINT_RE.search(msg))
    hate_replied = bool(HATE_HINT_RE.search(rep)) if rep else False

    sexual_here = bool(SEXUAL_HINT_RE.search(msg))
    sexual_replied = bool(SEXUAL_HINT_RE.search(rep)) if rep else False

    spam_repeat = bool(spam_flags.get("repeat_spam"))
    emoji_flood = bool(spam_flags.get("emoji_flood_single"))

    # Contexto RAG enxuto
    rag_block = ""
    if contexts:
        rag_block = "\n\n".join(contexts[:5])

    # Regras de saída (bem rígidas pra parar “alucinação” e respostas genéricas)
    # Importante: você pediu "sem fallback" e "sem predefinidas".
    # Então a gente NÃO devolve "Entendido/Confirmado/Ignorado" salvo se fizer sentido real.
    system = f"""
Você é um moderador e assistente de servidor Discord.
Você decide UMA ação por vez: reply, no, ou mute.

Regras de personalidade:
- Seja curto, frio, direto e sempre com gramática.
- Não use emojis.
- Não diga que é robô, IA ou assistente.
- NÃO use respostas genéricas como "Entendido.", "Confirmado.", "Ignorado." sem relação direta com o texto.
- Não puxe assunto e não faça perguntas de volta, a menos que o usuário tenha feito uma pergunta explícita.
- Se não for possível ajudar por motivo de conteúdo inadequado, responda de forma seca (ex: "Não vou te ajudar com isso.") ou aplique mute se for infração.
- Respostas podem ser maiores quando o usuário faz uma pergunta que exige explicação. Em conversa normal, mantenha curto.

Regras de moderação:
- Se houver infração na mensagem respondida (replied_text), e o usuário mencionou o bot como reply/denúncia, você DEVE considerar punir o autor da mensagem respondida (target="replied_user") mesmo que o texto do denunciante seja normal.
- Se houver infração no texto atual de quem marcou o bot, o alvo é target="author".
- Infrações graves: discurso de ódio, conteúdo sexual explícito, ameaças, calúnia sem evidência, assédio pesado, palavrões graves.
- Spam: 3 mensagens repetidas (ou mais) no mesmo canal em pouco tempo OU flood de 7+ emojis numa mensagem.
- Canal formal: se strict_channel=True e a mensagem atual do autor tiver gíria/emoji/caixa alta excessiva/falta de pontuação, isso pode ser infração ao ser marcado o bot.
- Se is_admin_author=True, obedeça pedidos diretos do admin para moderar (mute) quando fizer sentido.

Formato da saída: responda APENAS com JSON válido, seguindo este schema:
{{
  "mode": "reply" | "no" | "mute",
  "reply": "texto" (obrigatório se mode for reply ou no),
  "mute_minutes": número inteiro (obrigatório se mode for mute),
  "reason": "motivo curto" (obrigatório se mode for mute),
  "target": "author" | "replied_user" (obrigatório se mode for mute)
}}

Restrições do reply:
- Não mencione {author_display} dentro do texto a menos que seja necessário.
- Se precisar mencionar alguém, use o mention que já vem no input.
- Termine com pontuação normal. Não coloque ponto extra em cima de "!" ou "?".
"""

    user = {
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

1) Primeiro, avalie se existe infração no replied_text (se existir) e se isso exige mute do replied_user.
2) Depois, avalie se o message_text atual exige mute do author.
3) Se não houver infração, responda de forma coerente:
   - Se for pergunta: responda com explicação.
   - Se for conversa/afirmação: responda curto e relevante (não genérico).
4) Use mode="no" apenas quando a melhor resposta seca for "Não." ou algo igualmente seco.
"""
    }

    # Chamada ao modelo (chat.completions é estável)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.3,
        max_tokens=380,
    )

    txt = (resp.choices[0].message.content or "").strip()
    data = _json_load_safe(txt)

    # Se o modelo quebrar JSON, a gente força algo "seguro", mas sem "textinho genérico".
    # (É o único fallback mínimo pra não travar o bot.)
    if not isinstance(data, dict) or "mode" not in data:
        return {"mode": "no", "reply": "Não."}

    mode = data.get("mode")

    if mode == "mute":
        # Sanitização
        minutes = int(data.get("mute_minutes", 30))
        minutes = max(1, min(minutes, 1440))
        target = data.get("target", "author")
        if target not in ("author", "replied_user"):
            target = "author"
        reason = _clean(data.get("reason", "Violação de regras."))
        return {
            "mode": "mute",
            "mute_minutes": minutes,
            "reason": reason[:220],
            "target": target,
        }

    if mode in ("reply", "no"):
        reply = _clean(data.get("reply", "Não."))
        # Evita auto-mencionar o próprio bot
        reply = re.sub(rf"<@!?{bot_user_id}>", "", reply).strip()

        # Corrige pontuação final SEM estragar "!" ou "?"
        if reply.endswith(".."):
            while reply.endswith(".."):
                reply = reply[:-1]
        if reply and reply[-1] not in ".!?":
            reply += "."

        # Remove respostas proibidas genéricas quando não encaixam
        low = reply.lower()
        banned = ["entendido.", "confirmado.", "ignorado."]
        if low in banned:
            # Em vez de genérico, devolve um "no" seco
            return {"mode": "no", "reply": "Não."}

        return {"mode": mode, "reply": reply}

    return {"mode": "no", "reply": "Não."}
