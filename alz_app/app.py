import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

# Load your trained model
model = load_model("alzheimers_model_mobilenetv2.h5")
IMG_SIZE = 128

# Class labels (based on your model)
class_names = ['Alzheimer', 'MildDemented', 'Normal']

# Page config
st.set_page_config(page_title="Alzheimer's Disease Detector")
st.title("🧠 Alzheimer's Disease Detection App")
st.write("Upload an MRI image to predict Alzheimer's stage.")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

# Grad-CAM Function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Handle prediction
if uploaded_file is not None:
    # Preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show result
    st.image(img, caption="Uploaded MRI", use_column_width=True)
    st.success(f"🧠 Prediction: **{pred_class}** with {confidence:.2f}% confidence")

    # Grad-CAM Visualization
    if st.button("🔍 Show Grad-CAM"):
        last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)
