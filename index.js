import express from 'express';
import fs from 'fs';
import path from 'path';
import { Configuration, OpenAIApi } from 'openai';
import dotenv from 'dotenv';
dotenv.config();

const app = express();
const PORT = process.env.PORT || 10000;

app.use(express.json());

const openai = new OpenAIApi(new Configuration({
  apiKey: process.env.OPENAI_API_KEY
}));

// Util para ler todos os arquivos .txt da pasta
function lerTextosDaPasta() {
  const dir = path.join(process.cwd(), './');
  const arquivos = fs.readdirSync(dir).filter(arquivo => arquivo.endsWith('.txt'));
  let conteudoTotal = '';

  for (const arquivo of arquivos) {
    const texto = fs.readFileSync(path.join(dir, arquivo), 'utf-8');
    conteudoTotal += `\nArquivo: ${arquivo}\n${texto}\n`;
  }

  return conteudoTotal;
}

app.post('/openai', async (req, res) => {
  // Valida mensagem corretamente
  if (!req.body || typeof req.body.mensagem !== 'string' || req.body.mensagem.trim() === '') {
    return res.status(400).json({ erro: 'Mensagem ausente ou inválida.' });
  }

  const mensagem = req.body.mensagem;
  const jogador = req.body.jogador || 'Desconhecido';

  try {
    const baseTextos = lerTextosDaPasta();
    const prompt = `Você é a IA do Exército Brasileiro do Tevez. Utilize as informações abaixo para responder à pergunta:
${baseTextos}

Jogador: ${jogador}
Pergunta: ${mensagem}
Resposta:`;

    const resposta = await openai.createChatCompletion({
      model: 'gpt-5-nano',
      messages: [
        { role: 'system', content: 'Você é um assistente militar educado e direto.' },
        { role: 'user', content: prompt }
      ]
    });

    const respostaFinal = resposta.data.choices[0]?.message?.content?.trim() || 'Não foi possível gerar uma resposta.';
    return res.json({ resposta: respostaFinal });

  } catch (erro) {
    console.error('[ERRO IA]:', erro);
    return res.status(500).json({ erro: 'Erro interno ao processar a mensagem.' });
  }
});

app.listen(PORT, () => console.log(`\u2705 Servidor rodando na porta ${PORT}`));
