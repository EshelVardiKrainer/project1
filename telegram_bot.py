import os
import io
from PIL import Image, ImageDraw
import numpy as np
import face_recognition
from telegram import Update, ReplyKeyboardMarkup, InputMediaPhoto
from telegram.ext import (
    ApplicationBuilder, ContextTypes, CommandHandler,
    MessageHandler, filters
)
from dotenv import load_dotenv
from sklearn.decomposition import PCA

# Load token from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Custom keyboard
keyboard = ReplyKeyboardMarkup([
    ['Add face'],
    ['Recognize faces'],
    ['Reset faces'],
    ['Similar celebs'],
    ['Map']
], resize_keyboard=True)

# In-memory storage
known_face_encodings = []
known_face_names = []
known_face_images = []  # store images for mapping
user_state = {}
celeb_encodings = []  # (encoding, celeb_name, image_path)

# Paths
CELEB_DIR = "celebs"

# States
WAITING_FOR_IMAGE = 1
WAITING_FOR_NAME = 2
WAITING_FOR_RECOGNITION_IMAGE = 3
WAITING_FOR_CELEB_LOOKALIKE_IMAGE = 4

# Load celeb face encodings on startup
def load_celeb_faces():
    for celeb_name in os.listdir(CELEB_DIR):
        celeb_path = os.path.join(CELEB_DIR, celeb_name)
        if not os.path.isdir(celeb_path):
            continue
        for filename in os.listdir(celeb_path):
            image_path = os.path.join(celeb_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    celeb_encodings.append((encodings[0], celeb_name, image_path))
            except Exception as e:
                print(f"Error loading celeb image {image_path}: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", reply_markup=keyboard)
    user_state[update.effective_user.id] = None

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id

    if user_state.get(user_id) == WAITING_FOR_NAME:
        name = text.strip()
        encoding = context.user_data.get('new_face_encoding')
        image = context.user_data.get('new_face_image')
        if encoding is not None and image is not None:
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            known_face_images.append(image)
            await update.message.reply_text(
                f"Great. I will now remember this face as {name}.",
                reply_markup=keyboard
            )
        user_state[user_id] = None
        return

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
        known_face_images.clear()
        await update.message.reply_text("All saved faces have been reset.", reply_markup=keyboard)
        user_state[user_id] = None
        return

    elif text == 'Similar celebs':
        user_state[user_id] = WAITING_FOR_CELEB_LOOKALIKE_IMAGE
        await update.message.reply_text("Upload me a picture of a single person and I will find which celebs are similar to that person")
        return

    elif text == 'Map':
        await send_similarity_map(update)
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
        context.user_data['new_face_image'] = image
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

    elif state == WAITING_FOR_CELEB_LOOKALIKE_IMAGE:
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            await update.message.reply_text("I couldn't detect any face in the image.", reply_markup=keyboard)
            user_state[user_id] = None
            return

        query_encoding = encodings[0]
        best_distance = float('inf')
        best_match_name = None
        best_match_image_path = None

        for celeb_encoding, celeb_name, image_path in celeb_encodings:
            distance = np.linalg.norm(query_encoding - celeb_encoding)
            if distance < best_distance:
                best_distance = distance
                best_match_name = celeb_name
                best_match_image_path = image_path

        if best_match_name:
            with open(best_match_image_path, 'rb') as img_file:
                await update.message.reply_photo(
                    photo=img_file,
                    caption=f"The celeb that the person is most similar to is: {best_match_name.replace('_', ' ').title()}",
                    reply_markup=keyboard
                )
        else:
            await update.message.reply_text("I couldn't find a similar celeb.", reply_markup=keyboard)

        user_state[user_id] = None

async def send_similarity_map(update: Update):
    encodings = known_face_encodings + [c[0] for c in celeb_encodings]
    images = known_face_images + [face_recognition.load_image_file(c[2]) for c in celeb_encodings]
    labels = known_face_names + [c[1] for c in celeb_encodings]

    if len(encodings) < 2:
        await update.message.reply_text("I need at least 2 faces to generate a map.", reply_markup=keyboard)
        return

    # Reduce to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(encodings)
    coords -= coords.min(axis=0)
    coords /= coords.max(axis=0)
    coords *= 800  # canvas size

    # Draw
    canvas = Image.new('RGB', (850, 850), 'white')
    for (x, y), image, label in zip(coords, images, labels):
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue
        top, right, bottom, left = face_locations[0]
        face = Image.fromarray(image[top:bottom, left:right])
        face = face.resize((64, 64))
        canvas.paste(face, (int(x), int(y)))

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    await update.message.reply_photo(photo=output, caption="Here's a map of all known faces", reply_markup=keyboard)

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

if __name__ == '__main__':
    print("Loading celeb encodings...")
    load_celeb_faces()
    print("Bot is running...")
    app.run_polling()
