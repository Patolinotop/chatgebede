const express = require("express");
const axios = require("axios");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.options("*", cors()); // preflight

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Lê TODOS os .txt da raiz do projeto (onde ficam index.js e package.json)
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

// Middleware: lê o BODY manualmente (não depende de Content-Type)
app.use((req, res, next) => {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(req.method)) return next();

  let data = "";
  req.setEncoding("utf8");

  req.on("data", (chunk) => {
    data += chunk;

    // limite simples (2MB)
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
        // segue pro próximo parser
      }
    }

    // tenta x-www-form-urlencoded
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

// Debug: veja exatamente o que chega
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
        model: "gpt-5-nano",
        messages: [
          {
            role: "system",
            content:
              "Responda usando os documentos fornecidos. Se não encontrar nos documentos, diga que não encontrou."
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

    const respostaTexto =
      r.data?.choices?.[0]?.message?.content?.trim() || "Não consegui gerar resposta.";

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
