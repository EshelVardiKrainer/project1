# 🧠 Face Recognition & Image Captioning Telegram Bot 🤖

This Telegram bot uses face recognition and AI-based image captioning to offer a fun and interactive experience. You can add and recognize faces, find similar celebrities, generate a facial map, and get detailed image descriptions with known faces highlighted.

## 🚀 Features

- 👤 **Add Face** – Upload a photo and name it, so the bot can recognize it later.
- 🧠 **Recognize Faces** – Upload an image, and the bot will tell you who's in it.
- 🌟 **Similar Celebs** – Find out which celebrity your face resembles most.
- 🗺️ **Map** – View a visual map of all added faces and celebs based on facial similarity.
- 🖼️ **Describe Image** – Get a detailed description of an image, including recognized people.

## 🛠️ Tech Stack

- Python 🐍
- Telegram Bot API
- `face_recognition` for facial analysis
- Hugging Face's BLIP for image captioning
- Scikit-learn's PCA for face mapping
- Pillow for image processing

## 📦 Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. **Create a virtual environment and activate it:**
    python3 -m venv .venv
    source .venv/bin/activate
3. **Install dependencies:**
    pip install -r requirements.txt
4. **Set up environment variables: Create a .env file and add your Telegram bot token and Hugging Face API key:**
    BOT_TOKEN=your_telegram_bot_token
    HF_API_KEY=your_huggingface_api_key
5. **Run the bot:**
    python telegram_bot.py

## 📂 File Structure

project/
├── telegram_bot.py           # Main Telegram bot logic
├── image_analysis.py         # Caption generation and face recognition
├── requirements.txt
├── .env                      # Environment variables (not committed)
├── README.md
├── celeb_map.png             # PCA-based face map
├── celebs/                   # Directory of known celeb images
├── user_images.pkl           # Stored user face data
└── ...

