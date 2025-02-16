import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the CLIP model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

st.title("CLIP Image and Text Similarity")

uploaded_file = st.file_uploader("Choose an image...", type="[png, jpg, jpeg]")
query_text = st.text_input("Enter a query text...")

if uploaded_file and query_text:
    image = Image.open(uploaded_file)
    inputs = processor(text=[query_text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    similarity = outputs.logits_per_image.item()

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Similarity Score: {similarity:.4f}")
