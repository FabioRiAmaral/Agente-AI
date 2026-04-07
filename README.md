# Agente AI
Projeto inicialmente feito em conjunto com a instituição de ensino UEG - Itaberaí pelo aluno Fábio Ribeiro do Amaral Filho, um simples agente AI escrito totalmente em Python usando a biblioteca Hugging Face Transformers para gerar embeddings.

## Sobre o projeto
Um agente de AI simples que integra o bot do Telegram usando a biblioteca oficial do Telegram para fazer uma ponte entre o Bot API do Telegram e o codigo escrito em Python.

## Como utilizar o repositório para uso próprio

### Clonando o repositório do projeto
Dentro do terminal GitBash, atente-se para pasta que esta selecionada, se for a desejada apenas cole o comando `git clone https://github.com/FabioRiAmaral/Agente-AI.git`, isso criara uma pasta com o nome do repositório e com todos os arquivos que há nele.

### Instale as dependências
Com o terminal GitBash dentro do repositório clonado, instale as dependências usando o comando `pip install -r requirements.txt` oque pode demorar algum tempo. É recomendado também usar um ambiente virtual python devido a grande quantidade de bibliotecas usadas, isolando o projeto em um ambiente seguro que não interfere no sistema global. 

Usando o comando `python -m venv venv` você pode criar um ambiente virtual dentro da pasta atualmente selecionada, selecione o ambiente vitual com o comando `source venv/Scripts/activate` no GitBash e prossiga com a instalação das dependências.

### Baixe o modelo de IA utilizado
Para facilitar, fiz um script que baixa automaticamente o modelo de IA utilizado para que o codigo rode de maneira correta, basta digitar no terminal `python scripts/baixarModelosUsados.py`, o comando rodará o codigo dentro do arquivo `baixarModelosUsados.py` reponsavel por instalar o modelo na pasta.

### Registrando sua API_KEY
No projeto a API_KEY usada foi ocultado e registrada em uma pasta .env que não é exportada para o GitHub por motivos de segurança, portanto, após criar o seu bot pelo [@BotFather](https://core.telegram.org/bots/tutorial) do Telegram, crie um arquivo usando o comando `touch .env` no terminal, abra o arquivo criado e escreva dentro do arquivo `API_KEY='sua-API_KEY'`.

### Inicie o Bot
No terminal GitBash e dentro do ambiente virtual, rode o codigo com o comando `python src/main.py`, envie o PDF no seu Bot dentro do Telegram e faça as perguntas relacionadas ao PDF e você recebera respostas baseadas no PDF enviado! 


## Como funciona por trás do codigo

### O que é um computador fazendo "IA"?
No fundo, um computador só sabe fazer uma coisa: operar com números. Ele não entende palavras, não entende frases, não entende significado. Então quando queremos que ele "entenda" um texto, precisamos transformar esse texto em números de alguma forma, o problema é: como transformar o significado de uma palavra em número?

Imagine que você quer ensinar um computador que "cachorro" e "cão" significam a mesma coisa. Se você simplesmente atribuir números arbitrários:
```
cachorro = 1
cão      = 2
gato     = 3
```
O computador vê que cachorro e cão são vizinhos (1 e 2), mas isso é coincidência — você poderia ter atribuído qualquer número. Não há significado real nessa numeração. a solução é criar os **embeddings**.

Em vez de um número, cada palavra ou frase é representada por uma lista de centenas de números. Por exemplo:
```
"cachorro" → [0.2, -0.5, 0.8, 0.1, -0.3, ...]  (384 números no total)
"cão"      → [0.2, -0.4, 0.7, 0.1, -0.2, ...]  (384 números no total)
"nuvem"    → [-0.6, 0.9, -0.1, 0.7, 0.4, ...]  (384 números no total)
```
Esses números não foram escolhidos manualmente, eles foram aprendidos por um modelo treinado em bilhões de frases. O modelo aprendeu que palavras que aparecem nos mesmos contextos têm números parecidos.

Pense assim: se em mil livros a palavra "cachorro" e a palavra "cão" sempre aparecem em frases parecidas, "meu cachorro late", "meu cão late", o modelo aprende que elas devem ter números parecidos.

Mas porque 384 dimensões?

Imagine um mapa comum, com longitude e latitude, duas dimensões, você consegue localizar qualquer cidade do mundo com apenas dois números, agora imagine um "mapa de significados" com 384 dimensões, cada dimensão captura alguma característica abstrata da linguagem. Nenhuma dimensão tem um nome explícito, mas o conjunto delas consegue capturar coisas como:

- Se a palavra é relacionada a tecnologia ou natureza
- Se representa algo físico ou abstrato
- Se tem conotação positiva ou negativa
- Se pertence a um contexto científico ou coloquial

Nenhum humano definiu isso, o modelo descobriu essas dimensões sozinho ao ler bilhões de textos.

### Então... Como podemos medir se dois significados são parecidos?
Agora que cada frase é um ponto em um espaço de 384 dimensões, precisamos medir a distância entre dois pontos. A medida usada se chama similaridade cosseno, e ela mede o ângulo entre dois vetores, imagine duas lanternas apontando de um ponto central. Se apontam na mesma direção (ângulo 0°), são idênticas, similaridade 1. Se apontam em direções opostas (ângulo 180°), são completamente diferentes, similaridade -1. Se apontam em direções perpendiculares (ângulo 90°), não têm relação, similaridade 0.

Com isso em mente, agora é possível entender o código e entender como ele realmente funciona!

### O que o `pdf_handler.py` faz com isso em mente

O PDF inteiro não pode ser comparado de uma vez, seria como tentar localizar uma informação num livro de 500 páginas sem índice. O código divide o texto em pedaços de 500 caracteres chamados chunks.

O overlap (sobreposição de 50 caracteres) existe porque informações importantes frequentemente ficam na fronteira entre dois pedaços. Sem sobreposição:

```
Chunk 1: "...a RAM é uma memória"
Chunk 2: "volátil que perde dados ao desligar..."
```

Nenhum chunk contém a definição completa. Com sobreposição, o chunk 2 começa um pouco antes, capturando o final do chunk 1:

```
Chunk 2: "...é uma memória volátil que perde dados ao desligar..."
```

### Como funciona o `embeddings.py`

O modelo **`all-MiniLM-L6-v2`** é uma rede neural com 22 milhões de parâmetros, quando você passa um texto, ele processa cada palavra em contexto com as outras e gera um vetor de 384 números representando o significado da frase inteira.

O ``tokenizer`` converte o texto em números antes de entrar no modelo, cada palavra (ou parte de palavra) vira um número de índice:

```
RAM armazena dados" → [1234, 567, 89, 2]
```

O modelo não lê letras, lê só esses índices e a partir deles, ele gera os vetores de significado.

O ``mean pooling`` existe porque o modelo gera um vetor separado para cada token (cada número no índice). Uma frase de 6 palavras gera 6 vetores,precisamos de um vetor só para representar a frase. O pooling faz a média desses vetores, ignorando tokens de preenchimento que não são texto real.

A ``normalização`` no final garante que todos os vetores tenham o mesmo "comprimento" matemático (magnitude 1), sem isso, frases longas teriam vetores maiores só por terem mais palavras, distorcendo as comparações.

### O que o ``embeddings_store.py`` faz

O ChromaDB é um banco de dados especializado em vetores. Diferente de um banco de dados comum que busca por palavras exatas, ele busca por proximidade de significado semantico, a base funcional dos embeddings.

Quando você salva os chunks do PDF, cada chunk vira um vetor de 384 números e é armazenado. Quando você faz uma pergunta, ela também vira um vetor. O ChromaDB então calcula a similaridade cosseno entre o vetor da pergunta e todos os vetores armazenados, retornando os chunks mais próximos — os que falam sobre o mesmo assunto que a pergunta, mesmo que usando palavras completamente diferentes.

### O que o ``translator.py`` faz e por que é tão necessário

De forma simples o MarianMT (modelo utilizado) é um modelo especializado só em tradução. Ele usa uma arquitetura chamada encoder-decoder: o encoder lê a frase em português e a transforma numa representação interna compacta de todo o significado. O decoder então gera a tradução em inglês palavra por palavra, sempre consultando essa representação interna para garantir que o significado seja preservado.

O ``num_beams=4`` faz o decoder explorar 4 caminhos de tradução ao mesmo tempo, como um jogador de xadrez que pensa 4 jogadas possíveis antes de decidir. Ele escolhe o caminho que tem maior probabilidade acumulada — gerando traduções mais naturais do que simplesmente escolher sempre a palavra mais provável no momento, e devido todo esse processo infelizmente essa parte do código é o maior responsável pela demora nas respostas.

## O que o ``qaChain.py`` faz juntando tudo que foi construido

O pipeline completo funciona assim, passo a passo:

1. Sua pergunta em português entra no tradutor → vira inglês

2. A pergunta em inglês é convertida em vetor de 384 números pelo embedder

3. O ChromaDB compara esse vetor com todos os chunks do PDF e retorna os 5 mais similares — os trechos que mais provavelmente contêm a resposta

4. Esses 5 trechos são traduzidos para inglês

5. Um prompt é montado assim:

```
"Usando APENAS o contexto abaixo, responda detalhadamente:

Contexto: [os 5 trechos do PDF em inglês]

Pergunta: [sua pergunta em inglês]

Resposta:"
```

6. O flan-t5-large recebe esse prompt. Ele é um modelo de 780 milhões de parâmetros treinado para seguir instruções. A frase "using ONLY the context below" ativa um comportamento específico que ele aprendeu: usar só as informações fornecidas em vez de inventar respostas do próprio "conhecimento"

7. Ele gera a resposta em inglês token por token

8. A resposta é traduzida de volta para português pelo MarianMT

O resultado é uma resposta que veio do conteúdo real do seu PDF, processada por um modelo que entende instruções em inglês, apresentada para você em português.

### E como ocorre a integração com o Telegram

O Telegram disponibiliza uma API oficial, uma interface baseada em requisição HTTP, os Bots são criados de maneira simples via [@BotFather](https://core.telegram.org/bots/tutorial), ao criar um Bot pela interface do Telegram você recebe um token exclusivo que é usado pelo codigo para autenticar cada requisição nos servidores do Telegram. No projeto foi usado a biblioteca [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) que abstrai o envio das requisições HTTP para os endpoints do Bot API, passando token e o texto da mensagem ou o indetificador do chat.

Antes de tudo, você precisa entender por que eu decidi usar funções async, um bot de Telegram pode receber mensagens de várias pessoas ao mesmo tempo e se o código fosse normal (síncrono), enquanto o bot estivesse processando a pergunta de uma pessoa, todas as outras ficariam congeladas esperando. O async resolve isso permitindo que o Python "pause" uma tarefa enquanto espera algo demorado (como baixar um arquivo ou aguardar resposta do modelo) e vá atender outra pessoa nesse intervalo, o await é o ponto onde esse "pause" acontece.

E é por aqui que tudo se incia, e também é exatamente aqui que tudo acaba, onde todo esse percurso de código e lógica se colidem, ele cria o primeiro PDF que será tranformados em chunks pelo `pdf_handler.py`, e ao mesmo tempo é o único que coleta os dados fornecidos por `qaChain.py`.

Ao receber um PDF é onde toda cadeia do ``qaChain.py`` entra em ação: reseta o ChromaDB, extrai texto do PDF, converte em chunks, transforma em vetores de 384 dimensões e armazena tudo. Depois disso, com uma simples variavel que controla a entrada de pdf, a variavel pdf_indexed se torna True, sinalizando que o bot está pronto para responder perguntas.