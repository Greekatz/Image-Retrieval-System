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

def choose_model(model_type):
    if model_type == "CLIP":
        return CLIP()
    elif model_type == "VGG19":
        return VGG19()
    else:
        st.error(f"Unkowwn model: {model_type}")


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

# Load image file paths
if not os.path.exists(METADATA_PATH):
    st.error("Metadata file not found! Please run `embedding.py` first.")
    st.stop()

with open(METADATA_PATH, "r") as f:
    image_files = f.read().splitlines()

# Load FAISS index
if not os.path.exists(INDEX_PATH):
    st.error("FAISS index file not found! Please run `embedding.py` first.")
    st.stop()

index = faiss.read_index(INDEX_PATH)


def choose_model(model_name):
    """
    Dynamically selects and loads the specified model.

    :param model_name: Model name (either "CLIP" or "VGG19")
    :return: Selected model object
    """
    if model_name == "CLIP":
        return CLIP()
    elif model_name == "VGG19":
        return VGG19()
    else:
        st.error(f"Unknown model: {model_name}")
        st.stop()


def image_to_image(image, feature_extractor, k=10):
    """
    Perform image-to-image search using FAISS and the selected model.

    :param image: PIL Image object (query image)
    :param feature_extractor: Model's feature extractor function
    :param k: Number of top results to retrieve (default: 10)
    :return: List of retrieved image paths and their similarity scores
    """
    query_image = feature_extractor(image).cpu().numpy()
    distances, indices = index.search(query_image, k)

    # Ensure valid indices (avoid out-of-range errors)
    valid_indices = [idx for idx in indices[0] if idx < len(image_files)]

    nearest_images = [image_files[idx] for idx in valid_indices]
    nearest_distances = [distances[0][i] for i in range(len(valid_indices))]

    return nearest_images, nearest_distances

def text_to_image(image, feature_extractor, k=10):
    query_text = 


def main():
    st.set_page_config(page_title="Prethesis Image Retrieval", layout="wide")
    st.title("🔍 Image Retrieval ")
    st.write("Upload an image to find similar images from the dataset.")

    model_choice = st.selectbox("Select Model:", ["CLIP", "VGG19"])

    selected_model = choose_model(model_choice)


    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Query Image", use_container_width=True)

        # Convert to PIL image
        image = Image.open(uploaded_file).convert("RGB")

        # Perform image search
        with st.spinner(f"Searching for similar images using {model_choice}..."):
            start_time = time.time()
            nearest_images, distances = image_to_image(image, selected_model.feature_extractor)
            end_time = time.time()
            st.success(f"Search completed in {end_time - start_time:.2f} seconds")

        # Display results
        st.write(f"### 🔥 Top Matching Images ({model_choice}):")

        if len(nearest_images) == 0:
            st.warning("No matching images found.")
        else:
            cols = st.columns(min(len(nearest_images), 5))  # Adjust columns dynamically

            for i, img_path in enumerate(nearest_images):
                retrieved_image = Image.open(img_path)
                cols[i % len(cols)].image(retrieved_image, caption=f"Similarity: {distances[i]:.4f}",
                                          use_container_width=True)


if __name__ == "__main__":
    main()
