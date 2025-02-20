import time
import torch
import faiss
import streamlit as st
from PIL import Image
from models.CLIP import CLIP 
from models.VGG19 import VGG19

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
INDEX_PATH = "image_embeddings.faiss"
METADATA_PATH = "image_files.txt"

clip_model = CLIP()

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load image file paths
with open(METADATA_PATH, "r") as f:
    image_files = f.read().splitlines()


def image_to_image(image, feature_extractor, k=10):
    """
    Perform image-to-image search using FAISS and CLIP feature extraction.

    :param image: PIL Image object (query image)
    :param feature_extractor: model's image feature extractor function
    :param k: Number of top results to retrieve (default: 5)
    :return: List of retrieved image paths and their similarity scores
    """
    query_image = feature_extractor(image).cpu().numpy()
    distances, indices = index.search(query_image, k)
    nearest_images = [image_files[idx] for idx in indices[0]]
    return nearest_images, distances[0]


def main():
    st.set_page_config(page_title="Prethesis Image Retrieval", layout="wide")
    st.title("🔍 Image Retrieval ")
    st.write("Upload an image to find similar images from the dataset.")

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Query Image", use_container_width=True)

        # Convert to PIL image
        image = Image.open(uploaded_file).convert("RGB")

        # Perform image search
        with st.spinner("Searching for similar images..."):
            start_time = time.time()
            nearest_images, distances = image_to_image(image, clip_model.image_feature_extractor)
            end_time = time.time()
            st.success(f"Search completed in {end_time - start_time:.2f} seconds")

        # Display results
        st.write("### 🔥 Top Matching Images:")
        cols = st.columns(len(nearest_images))

        for i, img_path in enumerate(nearest_images):
            retrieved_image = Image.open(img_path)
            cols[i].image(retrieved_image, caption=f"Similarity: {distances[i]:.4f}", use_container_width=True)


if __name__ == "__main__":
    main()
