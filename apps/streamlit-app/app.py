import streamlit as st
import requests
from PIL import Image
import io
import base64

# FastAPI URL
API_URL = "http://classification-inference-service.classification-inference.svc.cluster.local:8000/predict"
API_INFO_URL = "http://classification-inference-service.classification-inference.svc.cluster.local:8000/info"

# Helper function to make predictions
def make_prediction(image_bytes):
    files = {'file': image_bytes}
    response = requests.post(API_URL, files=files)
    return response.json()
    
# Helper function to get model info
def get_model_info():
    response = requests.get(API_INFO_URL)
    return response.json()
# UI
st.title("Image Classification with ResNet50")


# Display Model Info Button
if st.button("Show Model Info"):
    with st.spinner("Fetching model information..."):
        try:
            model_info = get_model_info()
            st.subheader("Model Information")
            st.write(f"**Name**: {model_info['name']}")
            st.write(f"**Input Shape**: {model_info['input_shape']}")
            st.write("**Labels**:")
            st.write(", ".join(model_info["labels"][:20]) + " ... (and more)")  # Show first 20 labels
        except Exception as e:
            st.error(f"Failed to fetch model info: {str(e)}")


st.write("Upload an image to classify it using the ResNet50 model.")



uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes.seek(0)

    if st.button("Classify Image"):
        # Send the image to the FastAPI endpoint
        with st.spinner("Classifying..."):
            result = make_prediction(image_bytes.getvalue())

        if "predictions" in result:
            # Display results
            st.subheader("Predictions")
            for pred in result["predictions"]:
                st.write(f"{pred['label']}: {pred['confidence']:.2%}")
        else:
            st.error("Error in classification.")
