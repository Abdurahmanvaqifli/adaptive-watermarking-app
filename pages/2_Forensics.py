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
    "Upload one original image and one or more suspected images "
    "to detect possible tampering."
)

original_file = st.file_uploader(
    "Original Image",
    type=["png", "jpg", "jpeg"],
    key="original"
)

suspected_files = st.file_uploader(
    "Suspected / Tampered Images",
    type=["png", "jpg", "jpeg"],
    key="suspected",
    accept_multiple_files=True
)


def analyze_tampering(original_rgb, suspected_rgb):
    original_rgb = cv2.resize(original_rgb, (512, 512))
    suspected_rgb = cv2.resize(suspected_rgb, (512, 512))

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    suspected_gray = cv2.cvtColor(suspected_rgb, cv2.COLOR_RGB2GRAY)

    score, diff = ssim(
        original_gray,
        suspected_gray,
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
        suspected_rgb,
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

    localization = suspected_rgb.copy()
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

    return {
        "original": original_rgb,
        "suspected": suspected_rgb,
        "ssim": score,
        "diff": diff,
        "threshold": threshold,
        "heatmap": blended_heatmap,
        "localization": localization,
        "tamper_percentage": tamper_percentage,
        "authenticity_score": authenticity_score,
        "severity": severity,
        "authenticity_status": authenticity_status,
        "detected_regions": detected_regions,
        "largest_area": largest_area,
        "region_data": region_data
    }


if original_file and suspected_files:

    original = Image.open(original_file).convert("RGB")
    original = np.array(original)

    st.subheader("Batch Tamper Detection Summary")

    batch_results = []
    analysis_outputs = []

    progress_bar = st.progress(0)
    progress_text = st.empty()

    total_files = len(suspected_files)

    for idx, suspected_file in enumerate(suspected_files):

        progress_text.write(
            f"Processing image {idx + 1}/{total_files}: {suspected_file.name}"
        )

        suspected = Image.open(suspected_file).convert("RGB")
        suspected = np.array(suspected)

        result = analyze_tampering(original, suspected)

        batch_results.append({
            "Image": suspected_file.name,
            "SSIM": round(result["ssim"], 4),
            "Tampered Area (%)": round(result["tamper_percentage"], 4),
            "Authenticity Score (%)": round(result["authenticity_score"], 4),
            "Severity": result["severity"],
            "Status": result["authenticity_status"],
            "Detected Regions": result["detected_regions"],
            "Largest Region Area (px)": result["largest_area"]
        })

        analysis_outputs.append({
            "file_name": suspected_file.name,
            "result": result
        })

        progress_bar.progress((idx + 1) / total_files)

    progress_text.write("Batch forensic analysis completed.")

    batch_df = pd.DataFrame(batch_results)

    st.dataframe(
        batch_df,
        use_container_width=True
    )

    avg_ssim = batch_df["SSIM"].mean()
    avg_tamper = batch_df["Tampered Area (%)"].mean()
    avg_auth = batch_df["Authenticity Score (%)"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Average SSIM", f"{avg_ssim:.4f}")
    c2.metric("Average Tampered Area", f"{avg_tamper:.2f}%")
    c3.metric("Average Authenticity", f"{avg_auth:.2f}%")

    st.subheader("Authenticity Score Chart")

    chart_df = batch_df[
        ["Image", "Authenticity Score (%)"]
    ].set_index("Image")

    st.bar_chart(chart_df)

    st.subheader("Detailed Analysis")

    selected_image_name = st.selectbox(
        "Select image for detailed forensic view",
        [item["file_name"] for item in analysis_outputs]
    )

    selected_output = next(
        item for item in analysis_outputs
        if item["file_name"] == selected_image_name
    )

    result = selected_output["result"]

    st.subheader("Forensics Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("SSIM Similarity", f"{result['ssim']:.4f}")
    col2.metric("Tampered Area", f"{result['tamper_percentage']:.2f}%")
    col3.metric("Authenticity Score", f"{result['authenticity_score']:.2f}%")
    col4.metric("Severity", result["severity"])

    if result["authenticity_score"] >= 90:
        st.success(result["authenticity_status"])
    elif result["authenticity_score"] >= 70:
        st.warning(result["authenticity_status"])
    else:
        st.error(result["authenticity_status"])

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
            round(result["ssim"], 4),
            round(result["tamper_percentage"], 4),
            round(result["authenticity_score"], 4),
            result["severity"],
            result["authenticity_status"],
            result["detected_regions"],
            result["largest_area"]
        ]
    })

    st.dataframe(
        result_df,
        use_container_width=True
    )

    st.subheader("Image Comparison")

    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.image(
            result["original"],
            caption="Original Image"
        )

    with img_col2:
        st.image(
            result["suspected"],
            caption="Suspected Image"
        )

    with img_col3:
        st.image(
            result["localization"],
            caption="Tamper Localization"
        )

    st.subheader("Difference and Heatmap Analysis")

    map_col1, map_col2, map_col3 = st.columns(3)

    with map_col1:
        st.image(
            result["diff"],
            caption="SSIM Difference Map",
            clamp=True
        )

    with map_col2:
        st.image(
            result["threshold"],
            caption="Threshold Map",
            clamp=True
        )

    with map_col3:
        st.image(
            result["heatmap"],
            caption="Tamper Heatmap"
        )

    st.subheader("Tampered Region Statistics")

    if result["region_data"]:
        region_df = pd.DataFrame(result["region_data"])
        st.dataframe(
            region_df,
            use_container_width=True
        )
    else:
        st.info("No significant tampered regions were detected.")

else:
    st.info("Please upload one original image and one or more suspected images.")
