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

    authenticity_score = max(
        0,
        min(100, (score * 100) - tamper_percentage)
    )

    if tamper_percentage < 2:
        severity = "Low"
    elif tamper_percentage < 10:
        severity = "Medium"
    else:
        severity = "High"

    if authenticity_score >= 90:
        authenticity_status = "Authentic / Very Low Risk"
    elif authenticity_score >= 70:
        authenticity_status = "Suspicious / Needs Review"
    else:
        authenticity_status = "Likely Tampered"

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
    region_data = []

    region_id = 1

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

            cv2.putText(
                localization,
                str(region_id),
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            region_data.append({
                "Region": region_id,
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h,
                "Area (px)": round(area, 2),
                "Area (%)": round((area / total_pixels) * 100, 4)
            })

            region_id += 1

    detected_regions = len(region_data)

    if detected_regions > 0:
        largest_region = max(region_data, key=lambda r: r["Area (px)"])
        largest_area = largest_region["Area (px)"]
    else:
        largest_area = 0

    st.subheader("Forensics Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("SSIM Similarity", f"{score:.4f}")
    col2.metric("Tampered Area", f"{tamper_percentage:.2f}%")
    col3.metric("Authenticity Score", f"{authenticity_score:.2f}%")
    col4.metric("Severity", severity)

    if authenticity_score >= 90:
        st.success(authenticity_status)
    elif authenticity_score >= 70:
        st.warning(authenticity_status)
    else:
        st.error(authenticity_status)

    result_df = pd.DataFrame({
        "Metric": [
            "SSIM Similarity",
            "Tampered Area (%)",
            "Authenticity Score (%)",
            "Severity Level",
            "Authenticity Status",
            "Number of Detected Regions",
            "Largest Region Area (px)"
        ],
        "Value": [
            round(score, 4),
            round(tamper_percentage, 4),
            round(authenticity_score, 4),
            severity,
            authenticity_status,
            detected_regions,
            largest_area
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

    st.subheader("Tampered Region Statistics")

    if region_data:
        region_df = pd.DataFrame(region_data)
        st.dataframe(
            region_df,
            use_container_width=True
        )
    else:
        st.info("No significant tampered regions were detected.")

else:
    st.info("Please upload both original and suspected images.")
