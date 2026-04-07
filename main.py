import logging, os, sys
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv
from qaChain import LangChainPDFPipeline

# Para usar o bot você deve primeiramente baixar as bibliotecas listadas no requirements.txt
# Escreva "pip install -r requirements.txt" no terminal, é recomendado também criar um ambiente virtual antes de baixas as bibliotecas

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO,
  handlers=[
    logging.FileHandler(os.path.join('./data/', 'app.log')), # Log aparece no terminal e no app.log para debbugar erros melhor
    logging.StreamHandler(sys.stdout)
  ]
)

pipeline = LangChainPDFPipeline()
pdf_indexed = False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text(
    "Envie um PDF no chat que eu responderei suas perguntas sobre ele!"
  )

async def document(update: Update, context: ContextTypes.DEFAULT_TYPE):
  global pdf_indexed
  doc = update.message.document
  if doc.mime_type != "application/pdf":
    await update.message.reply_text("Apenos consigo ler PDF, envie um PDF!")
    return
  await update.message.reply_text("PDF recebido! BELINHA SABE TUDO esta lendo")

  file = await context.bot.get_file(doc.file_id)
  await file.download_to_drive("./data/pdfUsuario.pdf")
  pipeline.index_pdf("pdfUsuario.pdf")
  pdf_indexed = True
  await update.message.reply_text("BELINHA leu seu PDF, faça a sua pergunta!")

async def message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  global pdf_indexed
  if not pdf_indexed:
    await update.message.reply_text("BELINHA ainda não leu nenhum PDF, envie um PDF primeiro")
    return

  question = update.message.text
  await update.message.reply_text("BELINHA esta pensando em como te responder, aguarde...")
  
  try:
    answer = pipeline.qa_with_llm(question)
    await update.message.reply_text(answer)
  except Exception as e:
    logging.error(f"Erro ao processar pergunta: {e}")
    await update.message.reply_text("Ocorreu um erro ao processar sua pergunta, tente novamente")

load_dotenv()
api_key = os.getenv("API_KEY")

if __name__ == '__main__':
  application = ApplicationBuilder().token(api_key).build()
  
  application.add_handler(CommandHandler('start', start))
  application.add_handler(MessageHandler(filters.Document.ALL, document))
  application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message))
  
  application.run_polling()