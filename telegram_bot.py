import os
import io
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
from telegram import Update, ReplyKeyboardMarkup, InputMediaPhoto
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, filters, ConversationHandler
)
from dotenv import load_dotenv

# Load token from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Custom keyboard
keyboard = ReplyKeyboardMarkup([
    ['Add face'],
    ['Recognize faces'],
    ['Reset faces']
], resize_keyboard=True)

# In-memory storage
known_face_encodings = []
known_face_names = []
user_state = {}

# States
WAITING_FOR_IMAGE = 1
WAITING_FOR_NAME = 2
WAITING_FOR_RECOGNITION_IMAGE = 3

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", reply_markup=keyboard)
    user_state[update.effective_user.id] = None

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id

    # ✅ Check if we're waiting for a name
    if user_state.get(user_id) == WAITING_FOR_NAME:
        name = text.strip()
        encoding = context.user_data.get('new_face_encoding')
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            await update.message.reply_text(
                f"Great. I will now remember this face as {name}.",
                reply_markup=keyboard
            )
        user_state[user_id] = None
        return

    # ✅ Handle button presses
    if text == 'Add face':
        user_state[user_id] = WAITING_FOR_IMAGE
        await update.message.reply_text("Upload an image with a single face")
        return

    elif text == 'Recognize faces':
        user_state[user_id] = WAITING_FOR_RECOGNITION_IMAGE
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in this image")
        return

    elif text == 'Reset faces':
        known_face_encodings.clear()
        known_face_names.clear()
        await update.message.reply_text("All saved faces have been reset.", reply_markup=keyboard)
        user_state[user_id] = None
        return

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    image = face_recognition.load_image_file(io.BytesIO(photo_bytes))

    state = user_state.get(user_id)

    if state == WAITING_FOR_IMAGE:
        encodings = face_recognition.face_encodings(image)
        if len(encodings) != 1:
            await update.message.reply_text("Please upload an image with exactly one face.")
            return
        context.user_data['new_face_encoding'] = encodings[0]
        await update.message.reply_text("Great. What's the name of the person in this image?")
        user_state[user_id] = WAITING_FOR_NAME

    elif state == WAITING_FOR_RECOGNITION_IMAGE:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            await update.message.reply_text("I couldn't detect any faces in the image.", reply_markup=keyboard)
            user_state[user_id] = None
            return

        names = []
        for encoding in face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            if len(distances) == 0 or np.min(distances) > 0.5:
                names.append("Unknown")
            else:
                best_match = np.argmin(distances)
                names.append(known_face_names[best_match])

        # Draw boxes
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        for (top, right, bottom, left), name in zip(face_locations, names):
            draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
            draw.text((left, top - 10), name, fill="blue")

        output = io.BytesIO()
        pil_image.save(output, format="PNG")
        output.seek(0)

        await update.message.reply_photo(photo=output, caption=f"I found {len(names)} face(s): {', '.join(names)}", reply_markup=keyboard)
        user_state[user_id] = None


app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

if __name__ == '__main__':
    print("Bot is running...")
    app.run_polling()
