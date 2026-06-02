import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(
    page_title="Digital Image Forensics",
    layout="wide"
)

st.title("🔍 Digital Image Forensics")

st.write(
    "Upload an original image and a suspected image "
    "to detect possible tampering."
)

original_file = st.file_uploader(
    "Original Image",
    type=["png", "jpg", "jpeg"],
    key="original"
)

tampered_file = st.file_uploader(
    "Suspected Image",
    type=["png", "jpg", "jpeg"],
    key="tampered"
)

if original_file and tampered_file:

    original = Image.open(original_file).convert("RGB")
    tampered = Image.open(tampered_file).convert("RGB")

    original = np.array(original)
    tampered = np.array(tampered)

    original = cv2.resize(original, (512, 512))
    tampered = cv2.resize(tampered, (512, 512))

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2GRAY
    )

    tampered_gray = cv2.cvtColor(
        tampered,
        cv2.COLOR_RGB2GRAY
    )

    score, diff = ssim(
        original_gray,
        tampered_gray,
        full=True
    )

    diff = (diff * 255).astype("uint8")

    threshold = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    localization = tampered.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 40:

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                localization,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

    st.success(
        f"SSIM Similarity Score: {score:.4f}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(
            original,
            caption="Original Image"
        )

    with c2:
        st.image(
            tampered,
            caption="Suspected Image"
        )

    with c3:
        st.image(
            localization,
            caption="Tamper Localization"
        )

    st.subheader("Difference Map")

    st.image(
        diff,
        clamp=True
    )

    st.subheader("Threshold Map")

    st.image(
        threshold,
        clamp=True
    )
