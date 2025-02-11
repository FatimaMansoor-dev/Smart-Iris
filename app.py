import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from qdrant_client import QdrantClient
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import base64
from PIL import Image
import zipfile
import io 


# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Qdrant client setup
client = QdrantClient(
    url='https://5508e774-44a9-440a-b842-bb765409525a.us-east4-0.gcp.cloud.qdrant.io:6333',
    api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2NTQ5NDczfQ.olOk9Z7W2lyufEu83kH4eCdSo_Kpo_Japs01kN6EW4o'
)

GEMINI_API_KEY='AIzaSyDSpt85i5J7eHTdjpxYqFIpqEHILbFYIsw'

# from google import genai

# client = genai.Client(api_key=GEMINI_API_KEY)

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents="Explain how AI works",
# )

# print(response.text)

collection_name = 'iris_iamges'

# Load the ResNet50 model
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.eval()

def extract_left_iris(image):
    """Extracts the left iris using MediaPipe FaceMesh."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(rgb_image)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            LEFT_IRIS = [474, 475, 476, 477]  
            iris_points = [(int(face_landmarks.landmark[idx].x * image.shape[1]),
                            int(face_landmarks.landmark[idx].y * image.shape[0])) for idx in LEFT_IRIS]
            center_x, center_y = np.mean(iris_points, axis=0).astype(int)
            radius = int(np.linalg.norm(np.array(iris_points[0]) - np.array(iris_points[2])) / 2)
            mask = np.zeros_like(image)
            cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
            return cv2.bitwise_and(image, mask)
    return None

def extract_hough_iris(image):
    """Detects the iris using HoughCircles if no face is detected."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=20, maxRadius=80)
    if circles is not None:
        x, y, r = np.round(circles[0, :][0]).astype("int")
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        return cv2.bitwise_and(image, mask)
    return None

def image_to_vector_resnet(image_path):
    with Image.open(image_path).convert("RGB") as img:
        inputs = processor(img, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
        vector = output.logits.squeeze().numpy()
    return vector / np.linalg.norm(vector)

def find_most_similar_images(input_image_path):
    input_vector = image_to_vector_resnet(input_image_path)
    similar_records = client.search(
        collection_name=collection_name,
        query_vector=input_vector,
        limit=5
    )
    similar_images = []
    if similar_records:
        for record in similar_records:
            img_data = base64.b64decode(record.payload['base64'])
            
            similar_images.append(Image.open(io.BytesIO(img_data)))
    return similar_images

st.title("Iris Extraction and Similarity Search")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name
    image = cv2.imread(image_path)
    st.image(image, caption="Uploaded Image", channels="BGR")
    if st.button("Find Similar Irises"):
        left_iris = extract_left_iris(image)
        if left_iris is not None:
            save_path = "extracted_left_iris.png"
            cv2.imwrite(save_path, left_iris)
        else:
            left_iris = extract_hough_iris(image)
            if left_iris is not None:
                save_path = "extracted_hough_0.png"
                cv2.imwrite(save_path, left_iris)

        if left_iris is not None:
            similar_images = find_most_similar_images(save_path)
            if similar_images:
                st.write("Most similar iris images:")
                cols = st.columns(len(similar_images))  # Create columns dynamically
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for i, img in enumerate(similar_images):
                        img_bytes_io = io.BytesIO()
                        img.save(img_bytes_io, format="JPEG", quality=95)

                        img_bytes = img_bytes_io.getvalue()

                        # Save image to zip
                        zip_file.writestr(f"similar_iris_{i+1}.png", img_bytes)

                        # Display image
                        with cols[i]:
                            st.image(img, use_container_width=True)

                            # Individual download button
                            st.download_button(
                                label=f"Download Image {i+1}",
                                data=img_bytes,
                                file_name=f"similar_iris_{i+1}.png",
                                mime="image/png"
                            )

                zip_buffer.seek(0)

                # Download button for all images as ZIP
                st.download_button(
                    label="Download All Similar Images (ZIP)",
                    data=zip_buffer,
                    file_name="similar_iris_images.zip",
                    mime="application/zip"
                )
            else:
                st.write("No similar images found.")
        else:
            st.error("No iris detected.")