const express = require('express');
const axios = require('axios');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// Lê todos os arquivos .txt da raiz do projeto
function carregarTextos() {
    const arquivos = fs.readdirSync(__dirname)
        .filter(nome => nome.endsWith('.txt'));

    let conteudoTotal = '';
    for (const nome of arquivos) {
        const caminho = path.join(__dirname, nome);
        const texto = fs.readFileSync(caminho, 'utf-8');
        conteudoTotal += `\n\n=== ${nome} ===\n${texto}`;
    }

    return conteudoTotal;
}

app.post('/openai', async (req, res) => {
    const { mensagem, jogador } = req.body;

    if (!mensagem) return res.status(400).json({ erro: 'Mensagem ausente.' });

    const textosBase = carregarTextos();

    try {
        const resposta = await axios.post(
            'https://api.openai.com/v1/chat/completions',
            {
                model: 'gpt-5-nano ',
                messages: [
                    { role: 'system', content: 'Você é um assistente de um exército no Roblox. Responda com base nos documentos fornecidos.' },
                    { role: 'system', content: `Documentos:\n${textosBase}` },
                    { role: 'user', content: mensagem }
                ],
                temperature: 0.6
            },
            {
                headers: {
                    Authorization: `Bearer ${OPENAI_API_KEY}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        const respostaTexto = resposta.data.choices[0].message.content;
        res.json({ resposta: respostaTexto });

    } catch (err) {
        console.error(err.response?.data || err.message);
        res.status(500).json({ erro: 'Erro ao consultar a OpenAI.' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`✅ Servidor rodando na porta ${PORT}`);
});
