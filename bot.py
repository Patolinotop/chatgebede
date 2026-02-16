import os
import re
import time
import asyncio
import traceback
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any

import discord

from rag import search_context, context_is_relevant
from ai_policy import decide_action

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN ausente")

# Debug: liga logs de RX quando mencionarem o bot / reply
DEBUG_RX = os.getenv("DEBUG_RX", "1").strip() == "1"

# Configs
HISTORY_LIMIT = 5
MIN_TYPING_DELAY = 1.5
DECIDE_TIMEOUT = float(os.getenv("DECIDE_TIMEOUT", "25"))
BUSY_COOLDOWN = float(os.getenv("BUSY_COOLDOWN", "0.0"))
EMOJI_FLOOD_MIN = int(os.getenv("EMOJI_FLOOD_MIN", "7"))
REPEAT_SPAM_MIN = int(os.getenv("REPEAT_SPAM_MIN", "3"))
REPEAT_WINDOW_SEC = int(os.getenv("REPEAT_WINDOW_SEC", "30"))

ADMIN_IDS = set()
_admin_ids_raw = (os.getenv("ADMIN_IDS") or "").strip()
if _admin_ids_raw:
    for x in _admin_ids_raw.split(","):
        x = x.strip()
        if x.isdigit():
            ADMIN_IDS.add(int(x))

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True

bot = discord.Client(intents=intents)

_busy_lock = asyncio.Lock()
_last_done_ts = 0.0
_repeat_cache: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}

MENTION_RE = re.compile(r"<@!?\d+>")
URL_RE = re.compile(r"https?://\S+")
SPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    r"(" r"[\U0001F300-\U0001FAFF]" r"|[\u2600-\u26FF]" r"|[\u2700-\u27BF]" r")",
    flags=re.UNICODE
)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def limpar_texto(t: str) -> str:
    t = (t or "").replace("\u200b", "")
    return SPACE_RE.sub(" ", t).strip()

def remove_bot_mention(text: str, bot_id: int) -> str:
    if not text:
        return ""
    text = re.sub(rf"<@!?{bot_id}>", "", text).strip()
    return SPACE_RE.sub(" ", text).strip()

def normalize_for_spam(text: str) -> str:
    t = (text or "").lower().strip()
    t = URL_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    return SPACE_RE.sub(" ", t).strip()

def count_emojis(text: str) -> int:
    return len(EMOJI_RE.findall(text or ""))

def has_slang(text: str) -> bool:
    t = (text or "").lower()
    slang = ["eae", "ae", "blz", "vlw", "tmj", "kkk", "kkkk", "mano", "mto", "pq", "td", "tb", "vc", "vcs"]
    return any(s in t for s in slang)

def style_flags(text: str) -> Dict[str, bool]:
    t = (text or "")
    emojis = count_emojis(t) > 0
    slang = has_slang(t)
    caps = bool(re.search(r"\b[A-Z]{5,}\b", t))
    punctuation = bool(re.search(r"[.!?]", t))
    return {"emoji": emojis, "slang": slang, "caps": caps, "punctuation": punctuation}

def is_strict_channel(channel_name: str) -> bool:
    name = (channel_name or "").lower()
    allowed = ["geral", "chat", "chat-geral", "bate-papo", "conversa", "off-topic", "offtopic"]
    if any(a in name for a in allowed):
        return False
    keywords = ["militar", "gradu", "avis", "regras", "anuncios", "anúncios", "midia", "mídia", "media", "comunic", "admin", "staff", "mod"]
    if any(k in name for k in keywords):
        return True
    return True

def best_two_roles(member: discord.Member) -> List[str]:
    try:
        roles = [r for r in member.roles if r and r.name != "@everyone"]
        roles.sort(key=lambda r: r.position, reverse=True)
        return [r.name for r in roles[:2]]
    except Exception:
        return []

def is_admin_author(member: Optional[discord.Member]) -> bool:
    if not member:
        return False
    if member.id in ADMIN_IDS:
        return True
    perms = member.guild_permissions if hasattr(member, "guild_permissions") else None
    if perms and (perms.administrator or perms.manage_guild or perms.manage_messages or perms.moderate_members):
        return True
    return False

async def fetch_last_messages(channel: discord.abc.Messageable, user_id: int, limit: int) -> List[str]:
    out = []
    try:
        async for m in channel.history(limit=50):
            if m.author and m.author.id == user_id and m.content:
                out.append(limpar_texto(m.content))
                if len(out) >= limit:
                    break
    except Exception:
        pass
    out.reverse()
    return out

def spam_flags(channel_id: int, author_id: int, message_text: str) -> Dict[str, Any]:
    flags = {"repeat_spam": False, "emoji_flood_single": False, "emoji_flood_window": False}
    norm = normalize_for_spam(message_text)
    ts = time.time()

    if count_emojis(message_text) >= EMOJI_FLOOD_MIN:
        flags["emoji_flood_single"] = True

    key = (channel_id, author_id)
    arr = _repeat_cache.get(key, [])
    arr.append((ts, norm))
    arr = [(t, s) for (t, s) in arr if ts - t <= REPEAT_WINDOW_SEC]
    _repeat_cache[key] = arr

    if norm and sum(1 for (_, s) in arr if s == norm) >= REPEAT_SPAM_MIN:
        flags["repeat_spam"] = True

    return flags

async def member_is_muted(member: discord.Member) -> bool:
    try:
        until = getattr(member, "communication_disabled_until", None)
        if until is None:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return until > now_utc()
    except Exception:
        return False

async def try_timeout(member: discord.Member, minutes: int, reason: str) -> None:
    until = now_utc() + timedelta(minutes=minutes)
    await member.timeout(until, reason=reason)

async def try_delete_message(msg: discord.Message) -> bool:
    try:
        await msg.delete()
        return True
    except Exception:
        return False

async def _resolve_reference(message: discord.Message) -> Optional[discord.Message]:
    if not message.reference:
        return None
    if isinstance(message.reference.resolved, discord.Message):
        return message.reference.resolved
    try:
        if message.reference.message_id:
            return await message.channel.fetch_message(message.reference.message_id)
    except Exception:
        return None
    return None

def _is_mention(message: discord.Message) -> bool:
    if not bot.user:
        return False
    # mais confiável que mentioned_in em alguns cenários
    return any(u.id == bot.user.id for u in (message.mentions or []))

def _should_trigger(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    if not bot.user:
        return False

    if _is_mention(message):
        return True

    if message.reference and isinstance(message.reference.resolved, discord.Message):
        if message.reference.resolved.author and message.reference.resolved.author.id == bot.user.id:
            return True

    if message.reference and message.reference.message_id:
        if _is_mention(message):
            return True

    return False

def _log_channel_perms(message: discord.Message) -> None:
    try:
        if not message.guild or not bot.user:
            return
        me = message.guild.me or message.guild.get_member(bot.user.id)
        if not me:
            return
        perms = message.channel.permissions_for(me)
        print(
            "[BOT] Perms no canal:"
            f" view_channel={perms.view_channel}"
            f" read_message_history={perms.read_message_history}"
            f" send_messages={perms.send_messages}"
            f" moderate_members={perms.moderate_members}"
            f" manage_messages={perms.manage_messages}"
        )
    except Exception:
        pass

async def _handle_message(message: discord.Message) -> None:
    start = time.time()
    channel_name = getattr(message.channel, "name", "dm")
    strict = is_strict_channel(channel_name)

    author_member = message.guild.get_member(message.author.id) if message.guild else None
    roles_top2 = best_two_roles(author_member) if author_member else []
    admin_author = is_admin_author(author_member)

    replied_msg = await _resolve_reference(message)
    replied_user_mention = ""
    replied_user_display = None
    replied_text = ""
    implicit_reply = False

    if replied_msg and replied_msg.author and bot.user and replied_msg.author.id != bot.user.id:
        implicit_reply = True
        replied_user_display = str(replied_msg.author)
        replied_user_mention = replied_msg.author.mention
        replied_text = replied_msg.content or ""

    raw_text = limpar_texto(message.content or "")
    cleaned = remove_bot_mention(raw_text, bot.user.id) if bot.user else raw_text
    cleaned = limpar_texto(cleaned)

    print(f"[BOT] Trigger em #{channel_name} (strict={strict}) por {message.author} ({message.author.id})")
    print(f"[BOT] Texto limpo: {repr(cleaned)}")
    _log_channel_perms(message)

    if not cleaned and not implicit_reply:
        print("[BOT] Sem texto após remover mention.")
        return

    last5_author = await fetch_last_messages(message.channel, message.author.id, HISTORY_LIMIT)
    last5_other = []
    if implicit_reply and replied_msg and replied_msg.author:
        last5_other = await fetch_last_messages(message.channel, replied_msg.author.id, HISTORY_LIMIT)

    sflags = spam_flags(message.channel.id, message.author.id, raw_text)

    combined_for_style = cleaned
    if implicit_reply and replied_text:
        combined_for_style = f"{cleaned}\n{replied_text}".strip()
    st_flags = style_flags(combined_for_style)

    print(f"[BOT] Style flags: {st_flags}")
    print(f"[BOT] Spam flags: {sflags}")
    print(f"[BOT] Hist autor(5): {len(last5_author)} | outro(5): {len(last5_other)}")

    print("[BOT] Iniciando RAG...")
    query_for_rag = cleaned if cleaned else replied_text
    ctx = search_context(query_for_rag)
    rel = context_is_relevant(ctx, query_for_rag)
    best_sim = rel.get("best_sim", 0.0) if isinstance(rel, dict) else 0.0
    relevant = rel.get("relevant", False) if isinstance(rel, dict) else False
    ctx_chunks = ctx if isinstance(ctx, list) else []
    print(f"[BOT] RAG best_sim={best_sim:.3f} relevant={relevant} ctx={len(ctx_chunks)}")
    print(f"[BOT] admin_author={admin_author}")

    print("[BOT] Iniciando decide_action (OpenAI)...")
    action = await asyncio.wait_for(
        asyncio.to_thread(
            decide_action,
            message_text=cleaned,
            author_display=str(message.author),
            author_roles_top2=roles_top2,
            replied_user_display=replied_user_display,
            replied_text=replied_text,
            last5_author=last5_author,
            last5_other=last5_other,
            is_admin_author=admin_author,
            admin_targets=[],
            contexts=ctx_chunks,
            context_relevant=relevant,
            channel_name=channel_name,
            strict_channel=strict,
            style_flags=st_flags,
            spam_flags=sflags,
            implicit_reply=implicit_reply,
            author_raw_text=cleaned,
            author_mention=message.author.mention,
            replied_user_mention=replied_user_mention,
            bot_user_id=bot.user.id if bot.user else 0,
        ),
        timeout=DECIDE_TIMEOUT
    )

    print(f"[BOT] Ação: {action}")

    await asyncio.sleep(MIN_TYPING_DELAY)

    mode = (action or {}).get("mode", "reply")

    if mode == "mute":
        target = (action or {}).get("target", "author")
        minutes = int((action or {}).get("mute_minutes", 30))
        reason = (action or {}).get("reason", "Violação de regras.")

        target_member: Optional[discord.Member] = None
        target_message_to_delete: Optional[discord.Message] = None

        if target == "replied_user" and replied_msg and replied_msg.author and message.guild:
            target_member = message.guild.get_member(replied_msg.author.id)
            target_message_to_delete = replied_msg
        else:
            target_member = author_member
            target_message_to_delete = message

        if not target_member:
            print("[BOT] Não foi possível resolver membro alvo para mute.")
            return

        if await member_is_muted(target_member):
            try:
                await message.reply("Este usuario(a) já está mutado(a).", mention_author=False)
            except Exception:
                pass
            print("[BOT] Alvo já estava mutado.")
            return

        try:
            await try_timeout(target_member, minutes, reason)
            print(f"[BOT] Mutado: {target_member} por {minutes} min. Motivo: {reason}")
        except Exception:
            print("[BOT] Erro ao mutar:")
            traceback.print_exc()
            return

        if target_message_to_delete:
            deleted = await try_delete_message(target_message_to_delete)
            print(f"[BOT] Delete da msg do infrator: {deleted}")

        try:
            report = f"{target_member.mention}\nTempo: {minutes} minuto(s).\nMotivo: {reason}"
            await message.channel.send(report)
        except Exception:
            pass

        print(f"[BOT] Concluído em {time.time() - start:.2f}s (mute)")
        return

    if mode == "no":
        reply = (action or {}).get("reply", "Não.")
        try:
            async with message.channel.typing():
                await asyncio.sleep(0.2)
            await message.reply(reply, mention_author=False)
        except Exception:
            print("[BOT] Erro ao responder (no):")
            traceback.print_exc()
        print(f"[BOT] Concluído em {time.time() - start:.2f}s (no)")
        return

    reply = (action or {}).get("reply", "").strip()
    if not reply:
        print("[BOT] Reply vazio, ignorando.")
        return

    reply = re.sub(rf"<@!?{bot.user.id}>", "", reply).strip()

    try:
        async with message.channel.typing():
            await asyncio.sleep(0.2)
        await message.reply(reply, mention_author=False)
    except Exception:
        print("[BOT] Erro ao responder (reply):")
        traceback.print_exc()

    print(f"[BOT] Concluído em {time.time() - start:.2f}s (reply)")

async def _guarded_handle(message: discord.Message) -> None:
    global _last_done_ts

    if _busy_lock.locked():
        return

    if BUSY_COOLDOWN > 0 and (time.time() - _last_done_ts < BUSY_COOLDOWN):
        return

    async with _busy_lock:
        try:
            await _handle_message(message)
        finally:
            _last_done_ts = time.time()

@bot.event
async def on_ready():
    print(f"Logado como {bot.user}.")

@bot.event
async def on_message(message: discord.Message):
    try:
        # DEBUG: prova que o evento chegou quando mencionam ou reply
        if DEBUG_RX and bot.user:
            mentioned = _is_mention(message)
            has_ref = bool(message.reference and message.reference.message_id)
            if mentioned or has_ref:
                ch = getattr(message.channel, "name", "dm")
                print(f"[BOT] RX msg em #{ch} de {message.author} ({message.author.id}) mentioned={mentioned} ref={has_ref} content_len={len(message.content or '')}")
                _log_channel_perms(message)

        if not _should_trigger(message):
            return
        await _guarded_handle(message)
    except Exception:
        print("[BOT] ERRO em on_message:")
        traceback.print_exc()

_async_started = False

async def start_discord_bot():
    global _async_started
    if _async_started:
        return
    _async_started = True
    print("[BOT] start_discord_bot(): iniciando bot.start(...)")
    await bot.start(DISCORD_TOKEN)

async def stop_discord_bot():
    try:
        await bot.close()
    except Exception:
        pass

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
