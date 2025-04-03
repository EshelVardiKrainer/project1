import os
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from dotenv import load_dotenv

# Load your token from the .env file
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Create the custom keyboard
keyboard = ReplyKeyboardMarkup([['Hello', 'World', 'Telegram', 'Bot']], resize_keyboard=True)

# Define /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose a word:", reply_markup=keyboard)

# Define message response
async def reply_same(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    if user_text in ['Hello', 'World', 'Telegram', 'Bot']:
        await update.message.reply_text(user_text)

# Main bot function
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply_same))

    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
