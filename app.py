import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import PIL

# === Load model ===
model = load_model("mobilenetv2_final.keras")

# === Class mapping ===
class_mapping = {
    "0": "Normal",
    "1": "Scoliosis",
    "2": "Spondylolisthesis"
}

# === Prediction function ===
def predict_image(image: PIL.Image.Image):
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    label = class_mapping[str(class_index)]
    confidence = predictions[0][class_index]
    return label, confidence

# === Streamlit UI ===
st.title("Lumbar Spine Deformity Classifier")
st.write("Upload a spine X-ray image to classify it into Normal, Scoliosis, or Spondylolisthesis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Classifying..."):
        label, confidence = predict_image(image)
        st.success(f"Prediction: **{label}** ({confidence*100:.2f}% confidence)")
