const express = require("express");
const axios = require("axios");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();

app.use(cors());
app.use(express.json({ limit: "2mb" })); // parse JSON

// Se vier JSON inválido, em vez de cair mudo, responde explicando
app.use((err, req, res, next) => {
  if (err instanceof SyntaxError && err.status === 400 && "body" in err) {
    return res.status(400).json({ erro: "JSON inválido no corpo da requisição." });
  }
  next();
});

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.warn("⚠️ OPENAI_API_KEY não está definida nas variáveis de ambiente do Render.");
}

function carregarTextosDaRaiz() {
  const dir = process.cwd(); // raiz do projeto no Render
  const arquivos = fs.readdirSync(dir).filter((nome) => nome.toLowerCase().endsWith(".txt"));

  let conteudoTotal = "";
  for (const nome of arquivos) {
    const filePath = path.join(dir, nome);
    const texto = fs.readFileSync(filePath, "utf-8");
    conteudoTotal += `\n\n=== ${nome} ===\n${texto}`;
  }

  return conteudoTotal.trim();
}

app.get("/health", (req, res) => {
  res.json({ ok: true });
});

app.post("/openai", async (req, res) => {
  // validação forte
  const body = req.body || {};
  const mensagem = typeof body.mensagem === "string" ? body.mensagem.trim() : "";
  const jogador = typeof body.jogador === "string" ? body.jogador : "Desconhecido";

  if (!mensagem) {
    return res.status(400).json({ erro: "Mensagem ausente ou inválida." });
  }

  const textosBase = carregarTextosDaRaiz();

  try {
    const r = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-5-nano",
        messages: [
          {
            role: "system",
            content:
              "Você é um assistente dentro de um jogo Roblox. Responda de forma clara e objetiva usando os documentos fornecidos. Se algo não estiver nos documentos, diga que não encontrou."
          },
          {
            role: "system",
            content: textosBase ? `Documentos base:\n${textosBase}` : "Documentos base: (nenhum .txt encontrado na raiz do projeto)"
          },
          {
            role: "user",
            content: `Jogador: ${jogador}\nPergunta: ${mensagem}`
          }
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
