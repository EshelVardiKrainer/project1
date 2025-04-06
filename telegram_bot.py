import os
import io
import numpy as np
import joblib
import pickle
from PIL import Image, ImageDraw
import face_recognition
from telegram import Update, ReplyKeyboardMarkup
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

# Paths
CELEB_DIR = "celebs"
CELEB_MAP_PATH = "celeb_map.png"
PCA_MODEL_PATH = "pca_model.pkl"
USER_COORDS_PATH = "user_coords.npy"
USER_IMAGES_PATH = "user_images.pkl"

# In-memory
user_state = {}
known_face_names = []

# States
WAITING_FOR_IMAGE = 1
WAITING_FOR_NAME = 2
WAITING_FOR_RECOGNITION_IMAGE = 3
WAITING_FOR_CELEB_LOOKALIKE_IMAGE = 4

# Startup cache
celeb_encodings = []
celeb_coords = []
celeb_images = []
celeb_labels = []
pca_model = None

# Cache user face data
user_coords = []
user_images = []
user_names = []

# Load celeb data
def load_celeb_faces():
    global celeb_encodings, celeb_coords, celeb_images, celeb_labels, pca_model

    for celeb_name in os.listdir(CELEB_DIR):
        celeb_path = os.path.join(CELEB_DIR, celeb_name)
        if not os.path.isdir(celeb_path):
            continue
        for filename in os.listdir(celeb_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(celeb_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    celeb_encodings.append(encodings[0])
                    celeb_images.append(image)
                    celeb_labels.append(celeb_name)
            except Exception as e:
                print(f"Error loading celeb image {image_path}: {e}")

    pca_model = PCA(n_components=2)
    celeb_coords = pca_model.fit_transform(celeb_encodings)
    celeb_coords -= celeb_coords.min(axis=0)
    celeb_coords /= celeb_coords.max(axis=0)
    celeb_coords *= 800

    joblib.dump(pca_model, PCA_MODEL_PATH)
    with open("celeb_coords.npy", "wb") as f:
        np.save(f, celeb_coords)
    with open("celeb_images.pkl", "wb") as f:
        pickle.dump(celeb_images, f)
    with open("celeb_labels.pkl", "wb") as f:
        pickle.dump(celeb_labels, f)

    generate_celeb_map()

# Create celeb map image
def generate_celeb_map():
    canvas = Image.new('RGB', (850, 850), 'white')
    for (x, y), image in zip(celeb_coords, celeb_images):
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue
        top, right, bottom, left = face_locations[0]
        face = Image.fromarray(image[top:bottom, left:right])
        face = face.resize((64, 64))
        canvas.paste(face, (int(x), int(y)))
    canvas.save(CELEB_MAP_PATH)

# Save user face data
def save_user_face_data():
    with open(USER_COORDS_PATH, "wb") as f:
        np.save(f, np.array(user_coords))
    with open(USER_IMAGES_PATH, "wb") as f:
        pickle.dump(user_images, f)

# Load user face data
def load_user_face_data():
    global user_coords, user_images
    if os.path.exists(USER_COORDS_PATH):
        user_coords = np.load(USER_COORDS_PATH).tolist()
    if os.path.exists(USER_IMAGES_PATH):
        with open(USER_IMAGES_PATH, "rb") as f:
            user_images.extend(pickle.load(f))

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", reply_markup=keyboard)
    user_state[update.effective_user.id] = None

# Handle menu text
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id

    if user_state.get(user_id) == WAITING_FOR_NAME:
        name = text.strip()
        encoding = context.user_data.get('new_face_encoding')
        image = context.user_data.get('new_face_image')
        if encoding is not None and image is not None:
            coords = pca_model.transform([encoding])[0]
            coords -= coords.min()
            coords /= coords.max()
            coords *= 800
            user_coords.append(coords)
            user_images.append(image)
            known_face_names.append(name)
            save_user_face_data()
            await update.message.reply_text(f"Great. I will now remember this face as {name}.", reply_markup=keyboard)
        user_state[user_id] = None
        return

    if text == 'Add face':
        user_state[user_id] = WAITING_FOR_IMAGE
        await update.message.reply_text("Upload an image with a single face")
        return

    elif text == 'Map':
        await send_combined_map(update)
        return

# Handle user image
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

# Send final map
async def send_combined_map(update: Update):
    canvas = Image.open(CELEB_MAP_PATH).copy()
    for (x, y), image in zip(user_coords, user_images):
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
    await update.message.reply_photo(photo=output, caption="Here's a combined map of celebs and user faces", reply_markup=keyboard)

# App setup
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

if __name__ == '__main__':
    print("Loading celeb data and PCA...")
    load_celeb_faces()
    load_user_face_data()
    print("Bot is running...")
    app.run_polling()