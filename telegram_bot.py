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
from image_analysis import generate_caption, recognize_faces, image_to_base64
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load token from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Custom keyboard
keyboard = ReplyKeyboardMarkup([
    ['Add face'],
    ['Recognize faces'],
    ['Reset faces'],
    ['Similar celebs'],
    ['Map'],
    ['Describe image']
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
known_face_encodings = []

# States
WAITING_FOR_IMAGE = 1
WAITING_FOR_NAME = 2
WAITING_FOR_RECOGNITION_IMAGE = 3
WAITING_FOR_CELEB_LOOKALIKE_IMAGE = 4
WAITING_FOR_IMAGE_DESCRIPTION = 5

# Startup cache
celeb_data = []
celeb_coords = []
celeb_images = []
celeb_labels = []
pca_model = None
celeb_min = None
celeb_max = None

user_coords = []
user_images = []
user_names = []


def load_celeb_faces():
    global celeb_data, celeb_coords, celeb_images, celeb_labels, pca_model, celeb_min, celeb_max

    celeb_encodings = []

    for celeb_name in os.listdir(CELEB_DIR):
        celeb_path = os.path.join(CELEB_DIR, celeb_name)
        if not os.path.isdir(celeb_path):
            continue

        encodings = []
        paths = []
        images = []

        for filename in os.listdir(celeb_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            image_path = os.path.join(celeb_path, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    encodings.append(encoding[0])
                    paths.append(image_path)
                    images.append(image)
            except Exception as e:
                print(f"Error loading celeb image {image_path}: {e}")

        if encodings:
            mean_encoding = np.mean(encodings, axis=0)
            celeb_data.append({
                'name': celeb_name,
                'mean_encoding': mean_encoding,
                'photo_encodings': encodings,
                'image_paths': paths,
                'images': images
            })
            celeb_encodings.append(mean_encoding)
            celeb_labels.append(celeb_name)
            celeb_images.append(images[0])

    # PCA map based on means
    pca_model = PCA(n_components=2)
    celeb_coords = pca_model.fit_transform(celeb_encodings)

    # Normalize using min/max
    celeb_min = celeb_coords.min(axis=0)
    celeb_max = celeb_coords.max(axis=0)

    celeb_coords = (celeb_coords - celeb_min) / (celeb_max - celeb_min)
    celeb_coords *= 800

    joblib.dump(pca_model, PCA_MODEL_PATH)
    with open("celeb_coords.npy", "wb") as f:
        np.save(f, celeb_coords)
    with open("celeb_labels.pkl", "wb") as f:
        pickle.dump(celeb_labels, f)
    generate_face_map()


def generate_face_map():
    global celeb_coords
    canvas = Image.new('RGB', (850, 850), 'white')

    for celeb, coords in zip(celeb_data, celeb_coords):
        image = celeb['images'][0]
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue

        top, right, bottom, left = face_locations[0]
        face = Image.fromarray(image[top:bottom, left:right])
        face = face.resize((64, 64))

        x, y = coords
        x = max(0, min(int(x), canvas.width - 64))
        y = max(0, min(int(y), canvas.height - 64))

        canvas.paste(face, (x, y))

    canvas.save(CELEB_MAP_PATH)

def save_user_face_data():
    with open(USER_COORDS_PATH, "wb") as f:
        np.save(f, np.array(user_coords))
    with open(USER_IMAGES_PATH, "wb") as f:
        pickle.dump({
            "images": user_images,
            "encodings": known_face_encodings,
            "names": known_face_names,
        }, f)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose an option:", reply_markup=keyboard)
    user_state[update.effective_user.id] = None

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    user_id = update.effective_user.id

    if user_state.get(user_id) == WAITING_FOR_NAME:
        global celeb_min, celeb_max

        name = text.strip()
        encoding = context.user_data.get('new_face_encoding')
        image = context.user_data.get('new_face_image')

        if encoding is not None and image is not None:
            coords = pca_model.transform([encoding])[0]
            coords = (coords - celeb_min) / (celeb_max - celeb_min)
            coords *= 800

            print(f"User normalized coords: {coords}")
            print(f"Using celeb_min: {celeb_min}, celeb_max: {celeb_max}")

            user_coords.append(coords)
            user_images.append(image)
            known_face_names.append(name)
            known_face_encodings.append(encoding)
            save_user_face_data()

            await update.message.reply_text(f"Great. I will now remember this face as {name}.", reply_markup=keyboard)

        user_state[user_id] = None
        return

    if text == 'Add face':
        user_state[user_id] = WAITING_FOR_IMAGE
        await update.message.reply_text("Upload an image with a single face")
        return

    elif text == 'Similar celebs':
        user_state[user_id] = WAITING_FOR_CELEB_LOOKALIKE_IMAGE
        await update.message.reply_text("Upload me a picture of a single person and I will find which celebs are similar to that person")
        return

    elif text == 'Map':
        await send_combined_map(update)
        return

    elif text == 'Describe image':
        user_state[user_id] = WAITING_FOR_IMAGE_DESCRIPTION
        await update.message.reply_text("Upload an image and I will describe it for you.")
        return

    elif text == 'Recognize faces':
        user_state[user_id] = WAITING_FOR_RECOGNITION_IMAGE
        await update.message.reply_text("Upload an image with at least one face and I will recognize who is in this image.")
        return
    
    # Fallback if the command isn't recognized
    await update.message.reply_text("I didn’t understand that. Please choose an option from the menu:", reply_markup=keyboard)
    user_state[user_id] = None

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
        print(f"Using celeb_min: {celeb_min}, celeb_max: {celeb_max}")

    elif state == WAITING_FOR_RECOGNITION_IMAGE:
        path = f"/tmp/{update.message.photo[-1].file_id}.jpg"
        with open(path, "wb") as f:
            f.write(photo_bytes)

        image_np = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        pil_image = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_image)

        recognized_names = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if any(matches):
                best_match_index = face_distances.argmin()
                name = known_face_names[best_match_index]
                recognized_names.append(name)
                draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
                draw.text((left, top - 10), name, fill="blue")

        output = io.BytesIO()
        pil_image.save(output, format="JPEG")
        output.seek(0)

        if recognized_names:
            await update.message.reply_photo(
                photo=output,
                caption=f"I found {len(recognized_names)} face(s) in this image and the people are: {', '.join(recognized_names)}",
                reply_markup=keyboard
            )
        else:
            await update.message.reply_text("I don’t recognize anyone in this image.", reply_markup=keyboard)

        user_state[user_id] = None

    elif state == WAITING_FOR_IMAGE_DESCRIPTION:
        path = f"/tmp/{update.message.photo[-1].file_id}.jpg"
        with open(path, "wb") as f:
            f.write(photo_bytes)

        caption = generate_caption(path)
        names = recognize_faces(path, known_face_encodings, known_face_names)
        if names:
            name_str = ", ".join(names)
            full_caption = f"{caption}\n\nAlso, I recognized: {name_str}"
        else:
            full_caption = caption + "\n\nI didn’t recognize anyone."

        await update.message.reply_text(full_caption, reply_markup=keyboard)
        user_state[user_id] = None

    elif state == WAITING_FOR_CELEB_LOOKALIKE_IMAGE:
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            await update.message.reply_text("There are no faces in this photo.", reply_markup=keyboard)
            user_state[user_id] = None
            return

        if len(face_encodings) > 1:
            await update.message.reply_text("There are more than one face in this photo. Please upload an image with just one face.", reply_markup=keyboard)
            user_state[user_id] = None
            return

        query_encoding = face_encodings[0]

        # Find closest celeb mean
        best_match = None
        best_distance = float('inf')

        for celeb in celeb_data:
            distance = np.linalg.norm(query_encoding - celeb['mean_encoding'])
            if distance < best_distance:
                best_distance = distance
                best_match = celeb

        if best_match:
            best_photo_index = np.argmin([
                np.linalg.norm(query_encoding - e)
                for e in best_match['photo_encodings']
            ])
            best_path = best_match['image_paths'][best_photo_index]

            with open(best_path, 'rb') as img_file:
                await update.message.reply_photo(
                    photo=img_file,
                    caption=f"The celeb that the person is most similar to is: {best_match['name'].replace('_', ' ').title()}",
                    reply_markup=keyboard
                )
        else:
            await update.message.reply_text("I couldn't find a similar celeb.", reply_markup=keyboard)

        user_state[user_id] = None


async def send_combined_map(update: Update):
    canvas = Image.open(CELEB_MAP_PATH).copy()
    for (x, y), image in zip(user_coords, user_images):
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            continue
        top, right, bottom, left = face_locations[0]
        face = Image.fromarray(image[top:bottom, left:right])
        face = face.resize((64, 64))

        # Center the face around (x, y)
        paste_x = int(x) - 32
        paste_y = int(y) - 32

        # Ensure the paste position is within bounds
        paste_x = max(0, min(paste_x, canvas.width - 64))
        paste_y = max(0, min(paste_y, canvas.height - 64))

        canvas.paste(face, (paste_x, paste_y))

    output = io.BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    await update.message.reply_photo(photo=output, caption="Here's a combined map of celebs and user faces", reply_markup=keyboard)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

if __name__ == '__main__':
    print("Loading celeb data and PCA...")
    load_celeb_faces()
    print("Bot is running...")
    app.run_polling()