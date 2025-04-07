import base64
import face_recognition
import requests
from dotenv import load_dotenv
import os

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

def generate_caption(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/octet-stream"
    }

    api_url = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"

    response = requests.post(
        api_url,
        headers=headers,
        data=image_bytes  # not using 'files' anymore
    )

    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif "error" in result:
        return f"Error from HF API: {result['error']}"
    else:
        return "Unable to generate caption."



def recognize_faces(image_path, known_encodings, known_names):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if any(matches):
            best_match_index = face_distances.argmin()
            recognized.append(known_names[best_match_index])

    return recognized

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"
