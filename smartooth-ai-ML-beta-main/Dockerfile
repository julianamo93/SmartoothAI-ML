# Usando a imagem oficial do Python
FROM python:3.11-slim

# Definindo o diretório de trabalho
WORKDIR /app

# Copiando os arquivos necessários
COPY . .

# Instalando as dependências
RUN pip install -r requirements.txt

# Expondo a porta em que a aplicação irá rodar
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["python", "run.py"]
