# Agente AI
Projeto inicialmente feito em conjunto com a instituição de ensino UEG - Itaberaí pelo aluno Fábio Ribeiro do Amaral Filho, um simples agente AI escrito totalmente em Python usando a biblioteca Hugging Face Transformers para gerar embeddings.

## Sobre o projeto
Um agente de AI simples que integra o bot do Telegram usando a biblioteca oficial do Telegram para fazer uma ponte entre o Bot API do Telegram e o codigo escrito em Python.

## Como utilizar o repositório para uso próprio

### Clonando o repositório do projeto
Dentro do terminal GitBash, atente-se para pasta que esta selecionada, se for a desejada apenas cole o comando `git clone https://github.com/FabioRiAmaral/Agente-AI.git`, isso criara uma pasta com o nome do repositório e com todos os arquivos que há nele

### Instale as dependências
Com o terminal GitBash dentro do repositório clonado, instale as dependências usando o comando `pip install -r requirements.txt` oque pode demorar algum tempo, é recomendado também usar um ambiente virtual python devido a grande quantidade de bibliotecas usadas, isolando o projeto em um ambiente seguro que não interfere no sistema global (usando o comando `python -m venv venv` crie um ambiente virtual dentro da pasta atualmente selecionada, selecione o ambiente vitual com o comando `source venv/Scripts/activate` no GitBash e prossiga com a instalação das dependências)

### Baixe o modelo de IA utilizado
Para facilitar, fiz um script que baixa automaticamente o modelo de IA utilizado para que o codigo rode de maneira correta, basta digitar no terminal `python scripts/baixarModelosUsados.py`, o comando rodará o codigo dentro do arquivo `baixarModelosUsados.py` reponsavel por instalar o modelo na pasta 

### Registrando sua API_KEY
No projeto a API_KEY usada foi ocultado e registrada em uma pasta .env que não é exportada para o GitHub por motivos de segurança, portanto, após criar o seu bot pelo [@BotFather](https://core.telegram.org/bots/tutorial) do Telegram, crie um arquivo usando o comando `touch .env` no terminal, abra o arquivo criado e escreva dentro do arquivo `API_KEY='sua-API_KEY'`

### Inicie o Bot
No terminal GitBash e dentro do ambiente virtual, rode o codigo com o comando `python src/main.py`, envie o PDF no seu Bot dentro do Telegram e faça as perguntas relacionadas ao PDF


## Como funciona por trás do codigo

### Como ocorre a integração Telegram
O Telegram disponibiliza uma API oficial, uma interface baseada em requisição HTTP, os Bots são criados de maneira simples via [@BotFather](https://core.telegram.org/bots/tutorial), ao criar um Bot pela interface do Telegram você recebe um token exclusivo que é usado pelo codigo para autenticar cada requisição nos servidores do Telegram. No projeto foi usado a biblioteca [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) que abstrai o envio das requisições HTTP para os endpoints do Bot API, passando token e o texto da mensagem ou o indetificador do chat. O codigo usado por aqui recebe um arquivo PDF pelo chat do Telegram e baixa para a pasta "data"