const express = require("express");
const axios = require("axios");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.options("*", cors()); // preflight

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.warn("⚠️ OPENAI_API_KEY não definida nas variáveis de ambiente do Render.");
}

// Mapeamento de versões (igual ao seu Python)
const MODEL_MAP = {
  "1": "gpt-4.1-nano",
  "1pro": "gpt-5-nano"
};

function carregarTextosDaRaiz() {
  const dir = process.cwd();
  const arquivos = fs.readdirSync(dir).filter((n) => n.toLowerCase().endsWith(".txt"));

  let conteudo = "";
  for (const nome of arquivos) {
    const fp = path.join(dir, nome);
    const texto = fs.readFileSync(fp, "utf-8");
    conteudo += `\n\n=== ${nome} ===\n${texto}`;
  }
  return conteudo.trim();
}

// Middleware: lê body manualmente (funciona mesmo se content-type vier zoado/vazio)
app.use((req, res, next) => {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(req.method)) return next();

  let data = "";
  req.setEncoding("utf8");

  req.on("data", (chunk) => {
    data += chunk;
    if (data.length > 2 * 1024 * 1024) {
      res.status(413).json({ erro: "Body grande demais (limite 2MB)." });
      req.destroy();
    }
  });

  req.on("end", () => {
    req.rawBody = data || "";
    req.parsedBody = {};

    const t = (data || "").trim();

    // tenta JSON
    if (t.startsWith("{") || t.startsWith("[")) {
      try {
        req.parsedBody = JSON.parse(data);
        return next();
      } catch {
        // segue
      }
    }

    // tenta urlencoded
    try {
      const params = new URLSearchParams(data);
      const obj = {};
      for (const [k, v] of params.entries()) obj[k] = v;
      req.parsedBody = obj;
    } catch {
      req.parsedBody = {};
    }

    next();
  });
});

// debug opcional
app.post("/echo", (req, res) => {
  res.json({
    ok: true,
    headers: req.headers,
    rawBodyPreview: (req.rawBody || "").slice(0, 500),
    parsedBody: req.parsedBody || {}
  });
});

app.get("/health", (req, res) => res.json({ ok: true }));

app.post("/openai", async (req, res) => {
  const body = req.parsedBody || {};

  // aceita "mensagem" OU "message"
  const mensagem = (
    typeof body.mensagem === "string" ? body.mensagem :
    typeof body.message === "string" ? body.message :
    ""
  ).trim();

  const jogador = (typeof body.jogador === "string" ? body.jogador : "Desconhecido");
  const version = (typeof body.version === "string" ? body.version : "1pro"); // default gpt-5-nano
  const selectedModel = MODEL_MAP[version] || "gpt-4.1-nano";

  if (!mensagem) {
    return res.status(400).json({
      erro: "Mensagem ausente ou inválida.",
      dica: "Envie JSON com { mensagem: \"...\" } (ou { message: \"...\" }).",
      parsedBody: body,
      rawBodyPreview: (req.rawBody || "").slice(0, 300)
    });
  }

  const textosBase = carregarTextosDaRaiz();

  try {
    // Chamada mínima (sem temperature) — evita o 400 que você viu
    const r = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: selectedModel,
        messages: [
          {
            role: "system",
            content:
              "Você é o ChatiGebede. Responda em Markdown. Se for código, use blocos ```linguagem. " +
              "Seja prestativo e profissional. Use os documentos fornecidos como base."
          },
          {
            role: "system",
            content: textosBase
              ? `Documentos:\n${textosBase}`
              : "Documentos: (nenhum .txt encontrado na raiz do projeto)"
          },
          { role: "user", content: mensagem }
        ]
      },
      {
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json"
        },
        timeout: 60000
      }
    );

    const reply = r.data?.choices?.[0]?.message?.content || "";
    return res.json({ resposta: reply, model_used: selectedModel });
  } catch (e) {
    const status = e.response?.status;
    const data = e.response?.data;

    console.error("❌ Erro OpenAI:", status, data || e.message);

    return res.status(500).json({
      erro: "Falha ao consultar a OpenAI.",
      detalhe: status ? `Status OpenAI: ${status}` : "Sem status",
      openai: data || undefined
    });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => console.log(`✅ Servidor rodando na porta ${PORT}`));
