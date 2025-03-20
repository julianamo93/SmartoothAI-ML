# SmarTooth AI

- Projeto: Smartooth AI - Challenge ODONTOPREV 2TDSZ-2024
  
- Integrantes ByteBloom Tech
  - RM554113 - Juliana Moreira da Silva - Desenvolvimento ML e DevOps
  - RM552590 - Kevin Nobre - Desenvolvedor Full Stack .NET e Mobile
  - RM552728 - Sabrina do Couto Xavier Lima - Desenvolvedora Full Stack Java e Mobile

- **Descrição do Projeto**
  - O SmarTooth AI é um sistema inteligente projetado para melhorar os serviços odontológicos, integrando inteligência artificial (AI) e aprendizado de máquina (ML) para oferecer uma experiência personalizada aos usuários. Este sistema oferece dicas personalizadas de saúde bucal, um filtro de procedimentos odontológicos com base nos planos de saúde, recomendações em tempo real, e um programa de recompensas para incentivar práticas de saúde bucal.

- **Índice**
  - Descrição do Projeto
  - Pré-requisitos
  - Instalação
  - Configuração do Ambiente Docker
  - Rotas da API
  - Instruções para Deploy
  - Exemplos de Chamadas de API
  - Contribuição
  - Pré-requisitos
  - Antes de começar, certifique-se de ter os seguintes requisitos instalados:

- **Docker**
  - Docker Compose
  - Instalação
  - Clone o repositório:

```bash
git clone https://github.com/seu-usuario/smartooth-ai.git
cd smartooth-ai
```
- Instale as dependências necessárias, caso esteja executando o projeto fora do Docker:

```bash
pip install -r requirements.txt
```
- **Configuração do Ambiente Docker**
  - Criação de Variáveis de Ambiente: Crie um arquivo .env na raiz do projeto com as configurações necessárias:

```bash
DATABASE_URL=<seu_banco_de_dados>
SECRET_KEY=<sua_chave_secreta>
```

- Build da Imagem Docker:

```bash
docker-compose build
```

- Iniciar o Contêiner:

```bash
docker-compose up -d
```

- **Rotas da API**
- Método	Endpoint	Descrição
- GET	/api/procedures	Lista todos os procedimentos odontológicos disponíveis.
- GET	/api/tips	Sugere dicas personalizadas de saúde bucal.
- POST	/api/rewards	Registra e consulta pontos de recompensa dos usuários.
- GET	/api/recommendations	Recomendações de cuidados personalizados.
- POST	/api/user/history	Registra histórico de saúde do paciente.

- **Instruções para Deploy**
  - Para executar a aplicação usando Docker e Docker Compose, siga as instruções abaixo:

- Build da Imagem:

```bash
docker-compose build
```

- Inicializar o Contêiner em Background:

```bash
docker-compose up -d
```

- Parar o Contêiner:

```bash
docker-compose down
```

- Acessando a Aplicação: A aplicação estará disponível em http://localhost:<PORTA>, com a porta configurada no docker-compose.yml.

- **Exemplos de Chamadas de API**
  - Para testar as rotas, você pode usar ferramentas como Postman ou cURL.

- Listar Procedimentos:

```bash
curl -X GET http://localhost:<PORTA>/api/procedures
```

- Obter Dicas de Saúde Bucal:

```bash
curl -X GET http://localhost:<PORTA>/api/tips
```

- Registrar Pontos de Recompensa:

```bash
curl -X POST http://localhost:<PORTA>/api/rewards -d '{"user_id": "123", "points": 10}' -H "Content-Type: application/json"
```

- Consultar Recomendações Personalizadas:

```bash
curl -X GET http://localhost:<PORTA>/api/recommendations?user_id=123
```

- **Contribuição**
Contribuições são bem-vindas! Para contribuir, siga os passos abaixo:

  - Faça um fork do repositório.
  - Crie uma branch com a nova feature: git checkout -b minha-feature.
  - Commit suas alterações: git commit -m 'Minha nova feature'.
  - Envie para a branch: git push origin minha-feature.
  - Abra um Pull Request.
