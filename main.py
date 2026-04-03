import logging, os, sys
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes, CommandHandler
from dotenv import load_dotenv


logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO,
  handlers=[
    logging.FileHandler(os.path.join('./data/', 'app.log')), # Log aparece no terminal e no app.log para debbugar erros melhor
    logging.StreamHandler(sys.stdout)
  ]
)

async def message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  user_text = update.message.text # Salva a mensagem enviada no telegran
  
  await context.bot.send_message( # envia uma mensagem de volta
    chat_id=update.effective_chat.id,
    text=update.message.text # A mensagem que volta é a mesma que foi enviada
    )
  
  with open("./data/userInput.txt", "w", encoding="utf-8") as file:
    file.write(user_text + "\n") 
      
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(
    chat_id=update.effective_chat.id,
    text="Testando o bot"
    )
  
async def document(update: Update, context: ContextTypes.DEFAULT_TYPE):
  document = update.message.document 
  
  if document.mime_type == "application/pdf": #Verifica se recebeu de fato um PDF
    await update.message.reply_text("PDF armazenado no banco de dados!")
    file = await context.bot.get_file(document.file_id) 
    await file.download_to_drive("./data\dadosUsuario.pdf") # Salva o PDF recebido na pasta de dados que serão utilizados
    
  else:
    await update.message.reply_text("Não foi recebido o documento do tipo PDF, tente novamente com um PDF")

load_dotenv()
api_key = os.getenv("API_KEY")

if __name__ == '__main__':
  application = ApplicationBuilder().token(api_key).build()
  
  start_handler = CommandHandler('start', start)
  document_handler = MessageHandler(filters.Document.ALL, document)
  message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), message)

  application.add_handler(document_handler)
  application.add_handler(message_handler)
  application.add_handler(start_handler)
  
  application.run_polling()