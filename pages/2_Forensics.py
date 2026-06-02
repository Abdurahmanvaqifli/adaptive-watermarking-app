import streamlit as st
import cv2
import numpy as np
import pandas as pd
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

    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_RGB2GRAY)

    score, diff = ssim(
        original_gray,
        tampered_gray,
        full=True
    )

    diff = (diff * 255).astype("uint8")

    inverse_diff = 255 - diff

    threshold = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    tampered_pixels = np.sum(threshold > 0)
    total_pixels = threshold.size
    tamper_percentage = (tampered_pixels / total_pixels) * 100

    if tamper_percentage < 2:
        severity = "Low"
    elif tamper_percentage < 10:
        severity = "Medium"
    else:
        severity = "High"

    heatmap = cv2.applyColorMap(
        inverse_diff,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    blended_heatmap = cv2.addWeighted(
        tampered,
        0.65,
        heatmap_rgb,
        0.35,
        0
    )

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

    st.subheader("Forensics Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("SSIM Similarity", f"{score:.4f}")
    col2.metric("Tampered Area", f"{tamper_percentage:.2f}%")
    col3.metric("Severity", severity)

    if severity == "Low":
        st.success("Low-level modification detected.")
    elif severity == "Medium":
        st.warning("Medium-level tampering detected.")
    else:
        st.error("High-level tampering detected.")

    result_df = pd.DataFrame({
        "Metric": [
            "SSIM Similarity",
            "Tampered Area (%)",
            "Severity Level",
            "Number of Detected Regions"
        ],
        "Value": [
            round(score, 4),
            round(tamper_percentage, 4),
            severity,
            len(contours)
        ]
    })

    st.dataframe(
        result_df,
        use_container_width=True
    )

    st.subheader("Image Comparison")

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

    st.subheader("Difference and Heatmap Analysis")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.image(
            diff,
            caption="SSIM Difference Map",
            clamp=True
        )

    with h2:
        st.image(
            threshold,
            caption="Threshold Map",
            clamp=True
        )

    with h3:
        st.image(
            blended_heatmap,
            caption="Tamper Heatmap"
        )

else:
    st.info("Please upload both original and suspected images.")
