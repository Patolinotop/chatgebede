import os, asyncio, re, traceback, time
from datetime import timedelta
from typing import List, Optional, Dict

import discord
from discord.ext import commands
from dotenv import load_dotenv

from rag import search_context, context_is_relevant
from ai_policy import decide_action

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
MIN_TYPING_EXTRA = float(os.getenv("MIN_TYPING_EXTRA", "1.5"))

# Debug controlado
DEBUG_ERRORS_IN_DISCORD = os.getenv("DEBUG_ERRORS_IN_DISCORD", "0") == "1"
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "25"))

ADMIN_NAMES = [x.strip().lower() for x in (os.getenv("ADMIN_NAMES", "")).split(",") if x.strip()]

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
_channel_locks: Dict[int, asyncio.Lock] = {}

def _log(msg: str):
    print(f"[BOT] {msg}", flush=True)

def get_lock(channel_id: int) -> asyncio.Lock:
    if channel_id not in _channel_locks:
        _channel_locks[channel_id] = asyncio.Lock()
    return _channel_locks[channel_id]

def display_name(user: discord.abc.User) -> str:
    gn = getattr(user, "global_name", None)
    return (gn or user.name or "Usuário").strip()

def top2_roles(member: discord.Member) -> List[str]:
    roles = [r for r in member.roles if r.name != "@everyone"]
    roles_sorted = sorted(roles, key=lambda r: r.position, reverse=True)
    return [r.name for r in roles_sorted[:2]]

def is_admin_by_name(member: discord.Member) -> bool:
    dn = display_name(member).lower()
    un = (member.name or "").lower()
    return (dn in ADMIN_NAMES) or (un in ADMIN_NAMES)

async def last_n_messages_from(channel: discord.TextChannel, user_id: int, n: int = 5) -> List[str]:
    msgs = []
    async for m in channel.history(limit=60, oldest_first=False):
        if m.author and m.author.id == user_id:
            txt = (m.content or "").strip()
            if txt:
                msgs.append(txt)
            if len(msgs) >= n:
                break
    return msgs

def triggered(message: discord.Message) -> bool:
    if message.author.bot:
        return False
    if bot.user is None:
        return False

    mentioned = bot.user in message.mentions
    replied_to_bot = False
    if message.reference and isinstance(message.reference.resolved, discord.Message):
        replied_to_bot = (message.reference.resolved.author.id == bot.user.id)

    return mentioned or replied_to_bot

def strip_bot_mention(text: str) -> str:
    if not bot.user:
        return (text or "").strip()
    t = re.sub(rf"<@!?{bot.user.id}>", "", (text or ""))
    return t.strip()

async def try_timeout(member: discord.Member, minutes: int, reason: str):
    until = discord.utils.utcnow() + timedelta(minutes=minutes)
    await member.edit(timeout=until, reason=reason)

async def _handle_message(message: discord.Message):
    if not isinstance(message.channel, discord.TextChannel):
        return
    if not triggered(message):
        return

    lock = get_lock(message.channel.id)
    if lock.locked():
        _log(f"Ignorando: lock ativo no canal {message.channel.id}")
        return

    async with lock:
        _log(f"Trigger em #{message.channel.name} por {display_name(message.author)} ({message.author.id})")
        t0 = time.time()

        async with message.channel.typing():
            await asyncio.sleep(MIN_TYPING_EXTRA)

            author = message.author
            if not isinstance(author, discord.Member):
                return

            author_text = strip_bot_mention(message.content or "")
            _log(f"Texto limpo: {author_text!r}")

            if not author_text:
                _log("Sem texto após remover mention. Encerrando.")
                return

            replied_user: Optional[discord.Member] = None
            replied_text: Optional[str] = None
            if message.reference and isinstance(message.reference.resolved, discord.Message):
                replied_msg = message.reference.resolved
                if isinstance(replied_msg.author, discord.Member):
                    replied_user = replied_msg.author
                replied_text = (replied_msg.content or "").strip()

            last5_author = await last_n_messages_from(message.channel, author.id, 5)
            last5_other = []
            if replied_user:
                last5_other = await last_n_messages_from(message.channel, replied_user.id, 5)

            _log(f"Hist autor(5): {len(last5_author)} | outro(5): {len(last5_other)}")

            # RAG pode demorar; não deixa travar infinito
            contexts, best_sim = await asyncio.wait_for(
                asyncio.to_thread(search_context, author_text),
                timeout=OPENAI_TIMEOUT_SEC
            )
            relevant = context_is_relevant(best_sim)
            _log(f"RAG best_sim={best_sim:.3f} relevant={relevant} ctx={len(contexts)}")

            admin_author = is_admin_by_name(author) or author.guild_permissions.administrator
            _log(f"admin_author={admin_author}")

            # decisão do modelo também pode demorar
            action = await asyncio.wait_for(
                asyncio.to_thread(
                    decide_action,
                    author_text,
                    display_name(author),
                    top2_roles(author),
                    display_name(replied_user) if replied_user else None,
                    replied_text,
                    last5_author,
                    last5_other,
                    admin_author,
                    [],
                    contexts,
                    relevant,
                ),
                timeout=OPENAI_TIMEOUT_SEC
            )

            _log(f"Ação: {action}")

            if action["mode"] == "mute":
                target = author
                if action.get("target") == "replied_user" and replied_user:
                    target = replied_user

                me = message.guild.me
                if me and me.guild_permissions.moderate_members:
                    await try_timeout(target, int(action["mute_minutes"]), action["reason"])
                    report = (
                        f"Tempo do mute: {int(action['mute_minutes'])} minuto(s).\n"
                        f"Usuário: <@{target.id}>.\n"
                        f"Motivo: {action['reason']}\n"
                    )
                    await message.channel.send(report)
                else:
                    await message.channel.send("Não.")
                _log(f"Concluído em {time.time()-t0:.2f}s (mute)")
                return

            reply_text = action.get("reply", "Não.")
            await message.reply(reply_text, mention_author=False)
            _log(f"Concluído em {time.time()-t0:.2f}s (reply)")

@bot.event
async def on_ready():
    _log(f"Logado como {bot.user}.")

@bot.event
async def on_message(message: discord.Message):
    try:
        await _handle_message(message)
    except asyncio.TimeoutError:
        _log("TIMEOUT em processamento (OpenAI/GitHub lento).")
        traceback.print_exc()
        # Sem fallback “bonzinho”: mostra erro se debug estiver ligado
        if DEBUG_ERRORS_IN_DISCORD:
            try:
                await message.reply("Erro: timeout.", mention_author=False)
            except Exception:
                pass
    except Exception:
        _log("ERRO em on_message:")
        traceback.print_exc()
        if DEBUG_ERRORS_IN_DISCORD:
            # manda o erro curto pra você ver no Discord
            try:
                await message.reply("Erro interno. Veja logs.", mention_author=False)
            except Exception:
                pass

async def start_discord_bot():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN ausente")
    _log("Iniciando bot.start()")
    await bot.start(DISCORD_TOKEN)
