# 📦 Requisitos (salve em requirements.txt):
# flask
# openai
# GitPython
# python-dotenv

from flask import Flask, request, jsonify
from git import Repo
import os
import openai
from dotenv import load_dotenv

# 🔒 Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPO_URL = os.getenv("REPO_URL")  # URL do repositório GitHub (ex: https://github.com/usuario/repositorio.git)
LOCAL_REPO_PATH = "repo"

# 🧠 Configurar OpenAI
openai.api_key = OPENAI_API_KEY

# 🔁 Clonar ou atualizar o repositório
if not os.path.exists(LOCAL_REPO_PATH):
    Repo.clone_from(REPO_URL, LOCAL_REPO_PATH)
else:
    repo = Repo(LOCAL_REPO_PATH)
    origin = repo.remotes.origin
    origin.pull()

# 🚀 Iniciar app Flask
app = Flask(__name__)

# 🧾 Função para ler todos os .txt
def ler_todos_txt():
    textos = []
    for root, _, files in os.walk(LOCAL_REPO_PATH):
        for file in files:
            if file.endswith(".txt"):
                caminho = os.path.join(root, file)
                with open(caminho, "r", encoding="utf-8") as f:
                    textos.append(f.read())
    return "\n".join(textos)

# 🧠 Gerar resposta com OpenAI
def gerar_resposta(input_usuario, contexto):
    prompt = f"Baseando-se no seguinte conteúdo:\n{contexto}\n\nGere uma resposta curta e gramatical para: '{input_usuario}'\nResposta curta (máx. 100 caracteres):"
    try:
        resposta = openai.ChatCompletion.create(
            model="gpt-5.2",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {str(e)}"

# 📬 Rota da API
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    entrada = data.get("input", "")
    if not entrada:
        return jsonify({"reply": "Nenhuma entrada recebida."}), 400

    contexto = ler_todos_txt()
    resposta = gerar_resposta(entrada, contexto)
    return jsonify({"reply": resposta})

# ▶️ Rodar localmente (modo dev)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# ✅ Exemplo de uso:
# POST em /api/chatbot com JSON: {"input": "o que devo escrever no menu de ajuda?"}
# Resposta: {"reply": "Dê instruções claras para o jogador."}

# 🔧 .env esperado:
# OPENAI_API_KEY=sua-chave-da-openai
# REPO_URL=https://github.com/seu-usuario/seu-repositorio
