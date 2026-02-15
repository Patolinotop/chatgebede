import os, asyncio, re
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

ADMIN_NAMES = [x.strip().lower() for x in (os.getenv("ADMIN_NAMES", "")).split(",") if x.strip()]

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Regra: se já está respondendo no canal, ignora
_channel_locks: Dict[int, asyncio.Lock] = {}

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

@bot.event
async def on_ready():
    print(f"Logado como {bot.user}.")

@bot.event
async def on_message(message: discord.Message):
    # só canal de texto em servidor
    if not isinstance(message.channel, discord.TextChannel):
        return

    if not triggered(message):
        return

    lock = get_lock(message.channel.id)
    if lock.locked():
        # regra: se já está respondendo, ignora e nem faz request
        return

    async with lock:
        async with message.channel.typing():
            # delay mínimo extra
            await asyncio.sleep(MIN_TYPING_EXTRA)

            author = message.author
            if not isinstance(author, discord.Member):
                return

            # texto “limpo” sem mention
            author_text = strip_bot_mention(message.content or "")
            if not author_text:
                return

            # se for reply, pega alvo e texto
            replied_msg: Optional[discord.Message] = None
            replied_user: Optional[discord.Member] = None
            replied_text: Optional[str] = None

            if message.reference and isinstance(message.reference.resolved, discord.Message):
                replied_msg = message.reference.resolved
                if isinstance(replied_msg.author, discord.Member):
                    replied_user = replied_msg.author
                replied_text = (replied_msg.content or "").strip()

            # últimas 5 do autor e do replied_user (se houver)
            last5_author = await last_n_messages_from(message.channel, author.id, 5)
            last5_other = []
            if replied_user:
                last5_other = await last_n_messages_from(message.channel, replied_user.id, 5)

            # RAG do GitHub
            contexts, best_sim = search_context(author_text)
            relevant = context_is_relevant(best_sim)

            # admin?
            admin_author = is_admin_by_name(author) or author.guild_permissions.administrator

            action = decide_action(
                message_text=author_text,
                author_display=display_name(author),
                author_roles_top2=top2_roles(author),
                replied_user_display=display_name(replied_user) if replied_user else None,
                replied_text=replied_text,
                last5_author=last5_author,
                last5_other=last5_other,
                is_admin_author=admin_author,
                admin_targets=[],  # se você for usar lista do txt, dá pra plugar depois
                contexts=contexts,
                context_relevant=relevant,
            )

            # executar ação
            if action["mode"] == "mute":
                target = author
                if action.get("target") == "replied_user" and replied_user:
                    target = replied_user

                me = message.guild.me
                if me and me.guild_permissions.moderate_members:
                    try:
                        await try_timeout(target, int(action["mute_minutes"]), action["reason"])
                        report = (
                            f"Tempo do mute: {int(action['mute_minutes'])} minuto(s).\n"
                            f"Usuário: <@{target.id}>.\n"
                            f"Motivo: {action['reason']}\n"
                        )
                        await message.channel.send(report)
                    except discord.Forbidden:
                        await message.channel.send("Não.")
                else:
                    await message.channel.send("Não.")
                return

            reply_text = action.get("reply", "Não.")
            await message.reply(reply_text, mention_author=False)

async def start_discord_bot():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN ausente")
    await bot.start(DISCORD_TOKEN)
