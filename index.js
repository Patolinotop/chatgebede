const express = require("express");
const axios = require("axios");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());

// Captura o body bruto de QUALQUER content-type
app.use(express.raw({ type: "*/*", limit: "2mb" }));

function parseBody(req) {
  const buf = req.body;
  if (!Buffer.isBuffer(buf) || buf.length === 0) {
    req.rawBody = "";
    req.parsedBody = {};
    return;
  }

  const text = buf.toString("utf8");
  req.rawBody = text;

  // tenta JSON mesmo sem content-type
  const trimmed = text.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      req.parsedBody = JSON.parse(text);
      return;
    } catch {
      req.parsedBody = {};
      return;
    }
  }

  // tenta x-www-form-urlencoded
  try {
    const params = new URLSearchParams(text);
    const obj = {};
    for (const [k, v] of params.entries()) obj[k] = v;
    req.parsedBody = obj;
  } catch {
    req.parsedBody = {};
  }
}

app.use((req, res, next) => {
  parseBody(req);
  next();
});

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

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

// Debug
app.post("/echo", (req, res) => {
  res.json({
    ok: true,
    method: req.method,
    url: req.originalUrl,
    headers: req.headers,
    rawBodyPreview: (req.rawBody || "").slice(0, 500),
    parsedBody: req.parsedBody || {}
  });
});

app.get("/health", (req, res) => res.json({ ok: true }));

app.post("/openai", async (req, res) => {
  const body = req.parsedBody || {};
  const mensagem = (typeof body.mensagem === "string" ? body.mensagem : "").trim();
  const jogador = (typeof body.jogador === "string" ? body.jogador : "Desconhecido");

  if (!mensagem) {
    return res.status(400).json({
      erro: "Mensagem ausente ou inválida.",
      contentType: req.headers["content-type"] || null,
      rawBodyPreview: (req.rawBody || "").slice(0, 300),
      parsedBody: body
    });
  }

  try {
    const textosBase = carregarTextosDaRaiz();

    const r = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content:
              "Responda usando os documentos fornecidos. Se não tiver informação nos documentos, diga que não encontrou."
          },
          {
            role: "system",
            content: textosBase
              ? `Documentos base:\n${textosBase}`
              : "Documentos base: (nenhum .txt encontrado na raiz do projeto)"
          },
          { role: "user", content: `Jogador: ${jogador}\nPergunta: ${mensagem}` }
        ],
        temperature: 0.4
      },
      {
        headers: {
          Authorization: `Bearer ${OPENAI_API_KEY}`,
          "Content-Type": "application/json"
        },
        timeout: 60000
      }
    );

    const respostaTexto = r.data?.choices?.[0]?.message?.content?.trim() || "Não consegui gerar resposta.";
    return res.json({ resposta: respostaTexto });
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
