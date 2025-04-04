import face_recognition
import numpy as np

# Load images
image1 = face_recognition.load_image_file("face1.png")
image2 = face_recognition.load_image_file("face2.png")
image3 = face_recognition.load_image_file("face3.png")

# Get face encodings (we expect exactly 1 face per image)
enc1 = face_recognition.face_encodings(image1)
enc2 = face_recognition.face_encodings(image2)
enc3 = face_recognition.face_encodings(image3)

if not enc1 or not enc2 or not enc3:
    print("❌ One or more faces could not be detected.")
else:
    face1 = enc1[0]
    face2 = enc2[0]
    face3 = enc3[0]

    # Calculate distances
    dist_1_2 = np.linalg.norm(face1 - face2)
    dist_1_3 = np.linalg.norm(face1 - face3)

    print(f"Distance from Image 1 to Image 2: {dist_1_2:.4f}")
    print(f"Distance from Image 1 to Image 3: {dist_1_3:.4f}")

    if dist_1_2 < dist_1_3:
        print("✅ Image 1 is more similar to Image 2")
    else:
        print("✅ Image 1 is more similar to Image 3")
