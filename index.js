const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

app.post('/openai', async (req, res) => {
    const { mensagem, jogador } = req.body;

    if (!mensagem) return res.status(400).json({ erro: 'Mensagem ausente.' });

    try {
        const resposta = await axios.post(
            'https://api.openai.com/v1/chat/completions',
            {
                model: 'gpt-3.5-turbo',
                messages: [
                    { role: 'system', content: 'Você é um assistente dentro de um jogo Roblox.' },
                    { role: 'user', content: mensagem }
                ],
                temperature: 0.7
            },
            {
                headers: {
                    'Authorization': `Bearer ${OPENAI_API_KEY}`,
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
    console.log(`Servidor rodando na porta ${PORT}`);
});
