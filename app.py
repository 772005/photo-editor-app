import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="ShadeShift Shift the Mood, Change the Story", layout="wide")

st.title("📸 ShadeShift-> Shift the Mood, Change the Story")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Filters
filter_options = ["Original", "Grayscale", "Sepia", "Cartoon", "Pencil Sketch", "Invert", "Blur", "Sharpen",]
selected_filter = st.selectbox("Choose a filter", filter_options)

# Brightness & Contrast sliders
brightness = st.slider("Brightness", -100, 100, 0)
contrast = st.slider("Contrast", -50, 50, 0)

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    buf = np.int16(input_img)
    buf = buf * (contrast/127 + 1) - contrast + brightness
    buf = np.clip(buf, 0, 255)
    return np.uint8(buf)

def apply_sepia(image):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def apply_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def apply_pencil(image):
    gray, sketch = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def apply_sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = np.array(img.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Apply filter
    if selected_filter == "Grayscale":
        edited_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edited_img = cv2.cvtColor(edited_img, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "Sepia":
        edited_img = apply_sepia(img)
    elif selected_filter == "Cartoon":
        edited_img = apply_cartoon(img)
    elif selected_filter == "Pencil Sketch":
        edited_img = apply_pencil(img)
        edited_img = cv2.cvtColor(edited_img, cv2.COLOR_GRAY2BGR)
    elif selected_filter == "Invert":
        edited_img = cv2.bitwise_not(img)
    elif selected_filter == "Blur":
        edited_img = cv2.GaussianBlur(img, (15, 15), 0)
    elif selected_filter == "Sharpen":
        edited_img = apply_sharpen(img)
    else:
        edited_img = img

    # Apply brightness/contrast
    edited_img = apply_brightness_contrast(edited_img, brightness, contrast)

    # Convert BGR → RGB for display
    edited_rgb = cv2.cvtColor(edited_img, cv2.COLOR_BGR2RGB)

    st.image(edited_rgb, channels="RGB", use_column_width=True)

    # Download button
    result = Image.fromarray(edited_rgb)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("💾 Download Edited Image", data=byte_im, file_name="edited.png", mime="image/png")

else:
    st.info("Upload an image to start editing.")
