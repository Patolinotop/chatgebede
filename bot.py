import os, asyncio, re, traceback, time, unicodedata
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

DEBUG_ERRORS_IN_DISCORD = os.getenv("DEBUG_ERRORS_IN_DISCORD", "0") == "1"
OPENAI_TIMEOUT_SEC = float(os.getenv("OPENAI_TIMEOUT_SEC", "25"))

ADMIN_NAMES = [x.strip().lower() for x in (os.getenv("ADMIN_NAMES", "")).split(",") if x.strip()]

# thresholds
EMOJI_FLOOD_SINGLE = int(os.getenv("EMOJI_FLOOD_SINGLE", "7"))      # ✅ mínimo 7 na mesma mensagem
EMOJI_FLOOD_WINDOW = int(os.getenv("EMOJI_FLOOD_WINDOW", "15"))     # soma das últimas 5 (fica)
REPEAT_SPAM_COUNT = int(os.getenv("REPEAT_SPAM_COUNT", "3"))        # 3 repetições

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

async def apply_timeout(member: discord.Member, minutes: int, reason: str):
    until = discord.utils.utcnow() + timedelta(minutes=minutes)
    if hasattr(member, "timeout"):
        await member.timeout(until, reason=reason)
        return
    await member.edit(communication_disabled_until=until, reason=reason)

def is_currently_muted(member: discord.Member) -> bool:
    """
    Discord timeout ativo:
    communication_disabled_until > agora
    """
    until = getattr(member, "communication_disabled_until", None)
    if not until:
        return False
    try:
        now = discord.utils.utcnow()
        return until > now
    except Exception:
        return False

# ---------- canal estrito + estilo ----------
RELAXED_CHANNEL_KEYWORDS = {"geral", "chat", "bate-papo", "batepapo", "conversa"}
STRICT_CHANNEL_KEYWORDS = {
    "militar", "graduad", "avisos", "anuncio", "anúncio", "regras", "midia", "mídia",
    "comunicados", "documentos", "ordens", "instrucao", "instrução", "relatorios", "relatório"
}

def is_strict_channel(name: str) -> bool:
    n = (name or "").strip().lower()
    if any(k in n for k in STRICT_CHANNEL_KEYWORDS):
        return True
    if any(k == n or k in n for k in RELAXED_CHANNEL_KEYWORDS):
        return False
    return True

SLANG = {"eae", "q", "pq", "fds", "fdp", "mano", "véi", "vei", "ta", "tá", "blz", "kkk", "kkkk"}

def count_unicode_emojis(s: str) -> int:
    return sum(1 for ch in s if unicodedata.category(ch) == "So")

def count_custom_emojis(s: str) -> int:
    return len(re.findall(r"<a?:\w+:\d+>", s))

def has_emoji(s: str) -> bool:
    return (count_unicode_emojis(s) + count_custom_emojis(s)) > 0

def bad_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 6:
        return False
    upp = sum(1 for c in letters if c.isupper())
    low = sum(1 for c in letters if c.islower())
    return (upp >= 0.8 * (upp + low))

def bad_punctuation(s: str) -> bool:
    t = s.strip()
    if len(t) < 4:
        return True
    if not re.search(r"[.!?]$", t):
        return True
    if re.search(r"([!?\.])\1{3,}", t):
        return True
    return False

def contains_slang(s: str) -> bool:
    w = set(re.findall(r"[a-zA-ZÀ-ÿ]+", s.lower()))
    return any(x in w for x in SLANG)

def compute_style_flags(text: str) -> Dict[str, bool]:
    return {
        "emoji": has_emoji(text),
        "slang": contains_slang(text),
        "caps": bad_caps(text),
        "punctuation": bad_punctuation(text),
    }

# ---------- spam / flood ----------
def normalize_for_repeat(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"<@!?\d+>", "", s)
    s = re.sub(r"[^\wÀ-ÿ ]+", "", s)
    return s.strip()

def repeat_count(current: str, last_msgs: List[str]) -> int:
    cur = normalize_for_repeat(current)
    if not cur:
        return 0
    all_msgs = [normalize_for_repeat(x) for x in ([current] + last_msgs)]
    return sum(1 for x in all_msgs if x == cur)

def emoji_count_text(s: str) -> int:
    return count_unicode_emojis(s) + count_custom_emojis(s)

def emoji_window_count(current: str, last_msgs: List[str]) -> int:
    total = emoji_count_text(current)
    for m in last_msgs[:5]:
        total += emoji_count_text(m)
    return total

def compute_spam_flags(current: str, last_msgs: List[str]) -> Dict[str, bool]:
    rep = repeat_count(current, last_msgs)
    e_single = emoji_count_text(current)
    e_window = emoji_window_count(current, last_msgs)
    return {
        "repeat_spam": rep >= REPEAT_SPAM_COUNT,
        "emoji_flood_single": e_single >= EMOJI_FLOOD_SINGLE,   # ✅ agora 7+
        "emoji_flood_window": e_window >= EMOJI_FLOOD_WINDOW,
        "repeat_count": rep,
        "emoji_single": e_single,
        "emoji_window": e_window,
    }

async def try_delete_message(msg: Optional[discord.Message], channel: discord.TextChannel):
    if not msg:
        return
    me = channel.guild.me
    if not (me and me.guild_permissions.manage_messages):
        _log("Sem permissão Manage Messages para deletar.")
        return
    try:
        await msg.delete()
        _log("Mensagem do infrator deletada.")
    except discord.Forbidden:
        _log("Forbidden ao deletar (permissão/hierarquia).")
    except discord.NotFound:
        _log("Mensagem já não existe (NotFound).")
    except Exception:
        _log("Erro desconhecido ao deletar mensagem:")
        traceback.print_exc()

# ---------- handler ----------
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
        ch_name = message.channel.name or ""
        strict = is_strict_channel(ch_name)

        _log(f"Trigger em #{ch_name} (strict={strict}) por {display_name(message.author)} ({message.author.id})")
        t0 = time.time()

        async with message.channel.typing():
            await asyncio.sleep(MIN_TYPING_EXTRA)

            author = message.author
            if not isinstance(author, discord.Member):
                return

            # reply context
            replied_user: Optional[discord.Member] = None
            replied_text: Optional[str] = None
            replied_msg_obj: Optional[discord.Message] = None

            if message.reference and isinstance(message.reference.resolved, discord.Message):
                replied_msg_obj = message.reference.resolved
                if isinstance(replied_msg_obj.author, discord.Member):
                    replied_user = replied_msg_obj.author
                replied_text = (replied_msg_obj.content or "").strip()

            # author text
            author_text = strip_bot_mention(message.content or "")
            _log(f"Texto limpo: {author_text!r}")

            # se só menção e reply existe -> comando implícito
            if not author_text and replied_text:
                author_text = "Analise a mensagem respondida."
                _log("Texto vazio após menção; usando comando implícito de análise do reply.")

            if not author_text:
                _log("Sem texto após remover mention.")
                return

            style_flags = compute_style_flags(author_text)
            _log(f"Style flags: {style_flags}")

            last5_author = await last_n_messages_from(message.channel, author.id, 5)
            last5_other = []
            if replied_user:
                last5_other = await last_n_messages_from(message.channel, replied_user.id, 5)

            _log(f"Hist autor(5): {len(last5_author)} | outro(5): {len(last5_other)}")

            spam_flags = compute_spam_flags(author_text, last5_author)
            _log(f"Spam flags: {spam_flags}")

            _log("Iniciando RAG...")
            contexts, best_sim = await asyncio.wait_for(
                asyncio.to_thread(search_context, author_text),
                timeout=OPENAI_TIMEOUT_SEC
            )
            relevant = context_is_relevant(best_sim)
            _log(f"RAG best_sim={best_sim:.3f} relevant={relevant} ctx={len(contexts)}")

            admin_author = is_admin_by_name(author) or author.guild_permissions.administrator
            _log(f"admin_author={admin_author}")

            _log("Iniciando decide_action (OpenAI)...")
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
                    ch_name,
                    strict,
                    style_flags,
                    spam_flags,
                ),
                timeout=OPENAI_TIMEOUT_SEC
            )
            _log(f"Ação: {action}")

            if action["mode"] == "mute":
                # define o alvo do mute
                target = author
                offender_msg = message  # se o infrator é o autor, apaga a mensagem atual

                if action.get("target") == "replied_user" and replied_user:
                    target = replied_user
                    offender_msg = replied_msg_obj  # ✅ apaga a mensagem do infrator (a respondida)

                # ✅ sempre tenta deletar a mensagem do infrator
                await try_delete_message(offender_msg, message.channel)

                # ✅ se já está mutado, não aplica outro
                if is_currently_muted(target):
                    await message.channel.send("Este usuario(a) já está mutado(a).")
                    _log("Alvo já estava mutado, não aplicou novo timeout.")
                    _log(f"Concluído em {time.time()-t0:.2f}s (already-muted)")
                    return

                me = message.guild.me
                if not (me and me.guild_permissions.moderate_members):
                    _log("Sem permissão Moderate Members.")
                    await message.channel.send("Não.")
                    return

                try:
                    await apply_timeout(target, int(action["mute_minutes"]), action["reason"])
                except discord.Forbidden:
                    _log("Forbidden ao tentar mutar (hierarquia/permissão).")
                    if DEBUG_ERRORS_IN_DISCORD:
                        await message.channel.send("Erro: sem permissão para mutar.")
                    return
                except Exception:
                    _log("Erro desconhecido ao mutar:")
                    traceback.print_exc()
                    if DEBUG_ERRORS_IN_DISCORD:
                        await message.channel.send("Erro ao aplicar mute. Veja logs.")
                    return

                report = (
                    f"Tempo do mute: {int(action['mute_minutes'])} minuto(s).\n"
                    f"Usuário: <@{target.id}>.\n"
                    f"Motivo: {action['reason']}\n"
                )
                await message.channel.send(report)
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
        if DEBUG_ERRORS_IN_DISCORD:
            try:
                await message.reply("Erro: timeout.", mention_author=False)
            except Exception:
                pass
    except Exception:
        _log("ERRO em on_message:")
        traceback.print_exc()
        if DEBUG_ERRORS_IN_DISCORD:
            try:
                await message.reply("Erro interno. Veja logs.", mention_author=False)
            except Exception:
                pass

async def start_discord_bot():
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN ausente")
    _log("Iniciando bot.start()")
    await bot.start(DISCORD_TOKEN)
