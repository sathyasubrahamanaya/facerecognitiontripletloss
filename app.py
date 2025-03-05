# app.py
import streamlit as st
import numpy as np
import torch
from PIL import Image
import os
import json
import pandas as pd
from facenet_pytorch import MTCNN
from model import load_embedding_model, compute_embedding

# Set page configuration
st.set_page_config(page_title="One-Shot Face Recognition", layout="wide")
st.title("One-Shot Face Recognition App")
st.markdown("""
This production-grade app allows you to:
- **Enroll Faces:** Upload one or more images to enroll a person.
- **Recognize Face:** Upload an image to recognize the face.
- **View Enrollments:** Review the stored enrollment records.
""")

# Constants and settings
IMG_SIZE = (160, 160)
THRESHOLD = 0.8  # Adjust this threshold based on your model's performance
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CSV_FILE = "enrollments.csv"
UPLOAD_FOLDER = "enrolled_faces"

# Ensure directories exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["name", "file_path", "embedding"]).to_csv(CSV_FILE, index=False)

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=IMG_SIZE, device=DEVICE)

def detect_and_crop_face(image: Image.Image) -> Image.Image:
    """
    Detects the largest face using MTCNN and returns a cropped face resized to IMG_SIZE.
    Ensures that bounding box coordinates are converted to integers.
    Returns None if no face is detected.
    """
    image_np = np.array(image.convert("RGB"))
    boxes, _ = mtcnn.detect(image_np)
    if boxes is None:
        return None
    try:
        # Expecting boxes[0] as [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, boxes[0])
    except Exception as e:
        st.error(f"Error processing bounding box: {e}")
        return None
    x1, y1 = max(0, x1), max(0, y1)
    cropped = image_np[y1:y2, x1:x2]
    try:
        face_img = Image.fromarray(cropped)
    except Exception as e:
        st.error(f"Error converting cropped region to image: {e}")
        return None
    return face_img.resize(IMG_SIZE)

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses a PIL image for InceptionResnetV1:
    - Converts to RGB, resizes to IMG_SIZE,
    - Normalizes pixel values to [-1,1],
    - Converts to a torch tensor of shape (1, 3, H, W).
    """
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1,1]
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

# Load the face embedding model using model.py (cached for performance)
@st.cache(allow_output_mutation=True)
def get_embedding_model():
    return load_embedding_model(model_path="resnet_face_triplet.pth", device=DEVICE)
embedding_model = get_embedding_model()

# Functions for enrollment storage in CSV
def load_enrollments():
    """
    Loads enrollment records from CSV.
    Returns a dictionary mapping names to a list of embeddings.
    """
    df = pd.read_csv(CSV_FILE)
    gallery = {}
    for _, row in df.iterrows():
        name = row["name"]
        emb = np.array(json.loads(row["embedding"]))
        if name in gallery:
            gallery[name].append(emb)
        else:
            gallery[name] = [emb]
    return gallery

def append_enrollment_record(name, file_path, embedding):
    """
    Appends a new enrollment record (name, file path, embedding) to CSV.
    Embedding is stored as a JSON string.
    """
    record = {"name": name, "file_path": file_path, "embedding": json.dumps(embedding.tolist())}
    df = pd.DataFrame([record])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)

# Sidebar Navigation
menu = st.sidebar.selectbox("Menu", ["Enroll Face", "Recognize Face", "View Enrollments"])

if menu == "Enroll Face":
    st.header("Enroll a New Face")
    name = st.text_input("Enter the person's name")
    uploaded_files = st.file_uploader("Upload one or more face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if st.button("Enroll") and name and uploaded_files:
        enroll_count = 0
        for file in uploaded_files:
            try:
                image = Image.open(file)
                cropped_face = detect_and_crop_face(image)
                if cropped_face is None:
                    st.warning("No face detected in one of the images; skipping that image.")
                    continue
                # Save cropped face in a folder for the person
                person_folder = os.path.join(UPLOAD_FOLDER, name)
                if not os.path.exists(person_folder):
                    os.makedirs(person_folder)
                file_count = len(os.listdir(person_folder))
                file_path = os.path.join(person_folder, f"face_{file_count+1}.jpg")
                cropped_face.save(file_path)
                # Compute embedding
                face_tensor = preprocess_image(cropped_face)
                embedding = compute_embedding(face_tensor, embedding_model, device=DEVICE)
                append_enrollment_record(name, file_path, embedding)
                enroll_count += 1
                st.image(cropped_face, caption=f"Enrolled for {name}", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing an image: {e}")
        if enroll_count > 0:
            st.success(f"Successfully enrolled {enroll_count} face(s) for {name}.")

elif menu == "Recognize Face":
    st.header("Recognize a Face")
    uploaded_file = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            cropped_face = detect_and_crop_face(image)
            if cropped_face is None:
                st.error("No face detected in the image.")
            else:
                st.image(cropped_face, caption="Detected Face", use_column_width=True)
                face_tensor = preprocess_image(cropped_face)
                query_embedding = compute_embedding(face_tensor, embedding_model, device=DEVICE)
                gallery = load_enrollments()
                if not gallery:
                    st.error("No enrolled faces found. Please enroll a face first.")
                else:
                    distances = {}
                    for person, emb_list in gallery.items():
                        dists = [np.linalg.norm(query_embedding - emb) for emb in emb_list]
                        distances[person] = min(dists)
                    best_match = min(distances, key=distances.get)
                    best_distance = distances[best_match]
                    if best_distance < THRESHOLD:
                        st.success(f"Face recognized as **{best_match}** (distance: {best_distance:.4f})")
                    else:
                        st.error(f"No matching face found (best match: **{best_match}**, distance: {best_distance:.4f})")
        except Exception as e:
            st.error(f"Error during recognition: {e}")

elif menu == "View Enrollments":
    st.header("Enrolled Faces")
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            st.info("No enrollments found.")
        else:
            st.dataframe(df)
    else:
        st.info("No enrollments found. Please enroll a face first.")
