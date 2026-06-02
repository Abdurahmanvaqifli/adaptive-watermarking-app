import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import html
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(
    page_title="Digital Image Forensics",
    layout="wide"
)

# =========================
# LANGUAGE SYSTEM
# =========================

language = st.sidebar.selectbox(
    "🌐 Language / Dil",
    ["🇦🇿 Azərbaycan", "🇬🇧 English"]
)

lang = "az" if language == "🇦🇿 Azərbaycan" else "en"

T = {
    "az": {
        "title": "🔍 Rəqəmsal Şəkil Forensikası",
        "desc": "Mümkün təhrifləri aşkar etmək üçün bir orijinal şəkil və bir və ya bir neçə şübhəli şəkil yükləyin.",
        "original_image": "Orijinal şəkil",
        "suspected_images": "Şübhəli / dəyişdirilmiş şəkillər",
        "info_upload": "Zəhmət olmasa bir orijinal şəkil və bir və ya bir neçə şübhəli şəkil yükləyin.",
        "processing": "Emal olunur",
        "completed": "Toplu forensik analiz tamamlandı.",
        "batch_summary": "Toplu təhrif aşkarlanması nəticələri",
        "image": "Şəkil",
        "ssim": "SSIM",
        "tampered_area": "Dəyişdirilmiş sahə (%)",
        "authenticity_score": "Autentiklik göstəricisi (%)",
        "severity": "Risk səviyyəsi",
        "status": "Status",
        "detected_regions": "Aşkarlanmış regionlar",
        "largest_region": "Ən böyük region sahəsi (px)",
        "avg_ssim": "Orta SSIM",
        "avg_tampered": "Orta dəyişdirilmiş sahə",
        "avg_auth": "Orta autentiklik",
        "auth_chart": "Autentiklik göstəricisi qrafiki",
        "detailed_analysis": "Ətraflı analiz",
        "select_image": "Ətraflı forensik baxış üçün şəkil seçin",
        "forensics_summary": "Forensik xülasə",
        "ssim_similarity": "SSIM oxşarlığı",
        "tampered_area_metric": "Dəyişdirilmiş sahə",
        "authenticity_metric": "Autentiklik göstəricisi",
        "image_comparison": "Şəkillərin müqayisəsi",
        "suspected_image": "Şübhəli şəkil",
        "tamper_localization": "Təhrif lokalizasiyası",
        "diff_heatmap": "Fərq və istilik xəritəsi analizi",
        "difference_map": "SSIM fərq xəritəsi",
        "threshold_map": "Threshold xəritəsi",
        "tamper_heatmap": "Təhrif istilik xəritəsi",
        "region_stats": "Təhrif regionlarının statistikası",
        "no_regions": "Əhəmiyyətli təhrif regionu aşkar edilmədi.",
        "result_table": "Nəticə cədvəli",
        "metric": "Metrik",
        "value": "Dəyər",
        "html_report": "HTML hesabatı yüklə",
        "csv_report": "CSV nəticələrini yüklə",
        "report_section": "Hesabat ixracı",
        "low": "Aşağı",
        "medium": "Orta",
        "high": "Yüksək",
        "authentic": "Autentik / Çox aşağı risk",
        "suspicious": "Şübhəli / Yoxlama tələb olunur",
        "likely_tampered": "Böyük ehtimalla dəyişdirilib",
    },
    "en": {
        "title": "🔍 Digital Image Forensics",
        "desc": "Upload one original image and one or more suspected images to detect possible tampering.",
        "original_image": "Original Image",
        "suspected_images": "Suspected / Tampered Images",
        "info_upload": "Please upload one original image and one or more suspected images.",
        "processing": "Processing",
        "completed": "Batch forensic analysis completed.",
        "batch_summary": "Batch Tamper Detection Summary",
        "image": "Image",
        "ssim": "SSIM",
        "tampered_area": "Tampered Area (%)",
        "authenticity_score": "Authenticity Score (%)",
        "severity": "Severity",
        "status": "Status",
        "detected_regions": "Detected Regions",
        "largest_region": "Largest Region Area (px)",
        "avg_ssim": "Average SSIM",
        "avg_tampered": "Average Tampered Area",
        "avg_auth": "Average Authenticity",
        "auth_chart": "Authenticity Score Chart",
        "detailed_analysis": "Detailed Analysis",
        "select_image": "Select image for detailed forensic view",
        "forensics_summary": "Forensics Summary",
        "ssim_similarity": "SSIM Similarity",
        "tampered_area_metric": "Tampered Area",
        "authenticity_metric": "Authenticity Score",
        "image_comparison": "Image Comparison",
        "suspected_image": "Suspected Image",
        "tamper_localization": "Tamper Localization",
        "diff_heatmap": "Difference and Heatmap Analysis",
        "difference_map": "SSIM Difference Map",
        "threshold_map": "Threshold Map",
        "tamper_heatmap": "Tamper Heatmap",
        "region_stats": "Tampered Region Statistics",
        "no_regions": "No significant tampered regions were detected.",
        "result_table": "Result Table",
        "metric": "Metric",
        "value": "Value",
        "html_report": "Download HTML Report",
        "csv_report": "Download CSV Results",
        "report_section": "Report Export",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "authentic": "Authentic / Very Low Risk",
        "suspicious": "Suspicious / Needs Review",
        "likely_tampered": "Likely Tampered",
    }
}

t = T[lang]

st.title(t["title"])
st.write(t["desc"])


# =========================
# HELPER FUNCTIONS
# =========================

def get_severity(tamper_percentage):
    if tamper_percentage < 2:
        return "Low"
    elif tamper_percentage < 10:
        return "Medium"
    else:
        return "High"


def get_status(authenticity_score):
    if authenticity_score >= 90:
        return "Authentic / Very Low Risk"
    elif authenticity_score >= 70:
        return "Suspicious / Needs Review"
    else:
        return "Likely Tampered"


def translate_severity(severity):
    mapping = {
        "Low": t["low"],
        "Medium": t["medium"],
        "High": t["high"]
    }
    return mapping.get(severity, severity)


def translate_status(status):
    mapping = {
        "Authentic / Very Low Risk": t["authentic"],
        "Suspicious / Needs Review": t["suspicious"],
        "Likely Tampered": t["likely_tampered"]
    }
    return mapping.get(status, status)


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def generate_html_report(batch_df, selected_name, result, region_df=None):
    safe_title = html.escape(t["title"])
    safe_selected = html.escape(selected_name)

    batch_html = batch_df.to_html(index=False, escape=True)

    if region_df is not None and not region_df.empty:
        region_html = region_df.to_html(index=False, escape=True)
    else:
        region_html = f"<p>{html.escape(t['no_regions'])}</p>"

    report = f"""
    <!DOCTYPE html>
    <html lang="{lang}">
    <head>
        <meta charset="UTF-8">
        <title>{safe_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 35px;
                line-height: 1.5;
                color: #222;
            }}
            h1, h2 {{
                color: #1f4e79;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 12px;
                margin-bottom: 25px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .box {{
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-bottom: 15px;
                background-color: #fafafa;
            }}
        </style>
    </head>
    <body>
        <h1>{safe_title}</h1>

        <div class="box">
            <h2>{html.escape(t['detailed_analysis'])}</h2>
            <p><b>{html.escape(t['image'])}:</b> {safe_selected}</p>
            <p><b>{html.escape(t['ssim_similarity'])}:</b> {result['ssim']:.4f}</p>
            <p><b>{html.escape(t['tampered_area_metric'])}:</b> {result['tamper_percentage']:.2f}%</p>
            <p><b>{html.escape(t['authenticity_metric'])}:</b> {result['authenticity_score']:.2f}%</p>
            <p><b>{html.escape(t['severity'])}:</b> {html.escape(translate_severity(result['severity']))}</p>
            <p><b>{html.escape(t['status'])}:</b> {html.escape(translate_status(result['authenticity_status']))}</p>
            <p><b>{html.escape(t['detected_regions'])}:</b> {result['detected_regions']}</p>
            <p><b>{html.escape(t['largest_region'])}:</b> {result['largest_area']}</p>
        </div>

        <h2>{html.escape(t['batch_summary'])}</h2>
        {batch_html}

        <h2>{html.escape(t['region_stats'])}</h2>
        {region_html}
    </body>
    </html>
    """

    return report.encode("utf-8")
    # =========================
# TAMPER ANALYSIS
# =========================

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

    severity = get_severity(tamper_percentage)
    authenticity_status = get_status(authenticity_score)

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


# =========================
# FILE UPLOADERS
# =========================

original_file = st.file_uploader(
    t["original_image"],
    type=["png", "jpg", "jpeg"],
    key="original"
)

suspected_files = st.file_uploader(
    t["suspected_images"],
    type=["png", "jpg", "jpeg"],
    key="suspected",
    accept_multiple_files=True
)
# =========================
# MAIN PROCESS
# =========================

if original_file and suspected_files:

    original = Image.open(original_file).convert("RGB")
    original = np.array(original)

    st.subheader(t["batch_summary"])

    batch_results = []
    analysis_outputs = []

    progress_bar = st.progress(0)
    progress_text = st.empty()

    total_files = len(suspected_files)

    for idx, suspected_file in enumerate(suspected_files):

        progress_text.write(
            f"{t['processing']} {idx + 1}/{total_files}: {suspected_file.name}"
        )

        suspected = Image.open(suspected_file).convert("RGB")
        suspected = np.array(suspected)

        result = analyze_tampering(original, suspected)

        batch_results.append({
            t["image"]: suspected_file.name,
            t["ssim"]: round(result["ssim"], 4),
            t["tampered_area"]: round(result["tamper_percentage"], 4),
            t["authenticity_score"]: round(result["authenticity_score"], 4),
            t["severity"]: translate_severity(result["severity"]),
            t["status"]: translate_status(result["authenticity_status"]),
            t["detected_regions"]: result["detected_regions"],
            t["largest_region"]: result["largest_area"]
        })

        analysis_outputs.append({
            "file_name": suspected_file.name,
            "result": result
        })

        progress_bar.progress((idx + 1) / total_files)

    progress_text.write(t["completed"])

    batch_df = pd.DataFrame(batch_results)

    st.dataframe(
        batch_df,
        use_container_width=True
    )

    avg_ssim = batch_df[t["ssim"]].mean()
    avg_tamper = batch_df[t["tampered_area"]].mean()
    avg_auth = batch_df[t["authenticity_score"]].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(t["avg_ssim"], f"{avg_ssim:.4f}")
    c2.metric(t["avg_tampered"], f"{avg_tamper:.2f}%")
    c3.metric(t["avg_auth"], f"{avg_auth:.2f}%")

    st.subheader(t["auth_chart"])

    chart_df = batch_df[
        [t["image"], t["authenticity_score"]]
    ].set_index(t["image"])

    st.bar_chart(chart_df)

    st.subheader(t["detailed_analysis"])

    selected_image_name = st.selectbox(
        t["select_image"],
        [item["file_name"] for item in analysis_outputs]
    )

    selected_output = next(
        item for item in analysis_outputs
        if item["file_name"] == selected_image_name
    )

    result = selected_output["result"]

    st.subheader(t["forensics_summary"])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(t["ssim_similarity"], f"{result['ssim']:.4f}")
    col2.metric(t["tampered_area_metric"], f"{result['tamper_percentage']:.2f}%")
    col3.metric(t["authenticity_metric"], f"{result['authenticity_score']:.2f}%")
    col4.metric(t["severity"], translate_severity(result["severity"]))

    translated_status = translate_status(result["authenticity_status"])

    if result["authenticity_score"] >= 90:
        st.success(translated_status)
    elif result["authenticity_score"] >= 70:
        st.warning(translated_status)
    else:
        st.error(translated_status)

    result_df = pd.DataFrame({
        t["metric"]: [
            t["ssim_similarity"],
            t["tampered_area"],
            t["authenticity_score"],
            t["severity"],
            t["status"],
            t["detected_regions"],
            t["largest_region"]
        ],
        t["value"]: [
            round(result["ssim"], 4),
            round(result["tamper_percentage"], 4),
            round(result["authenticity_score"], 4),
            translate_severity(result["severity"]),
            translated_status,
            result["detected_regions"],
            result["largest_area"]
        ]
    })

    st.dataframe(
        result_df,
        use_container_width=True
    )

    st.subheader(t["image_comparison"])

    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.image(
            result["original"],
            caption=t["original_image"]
        )

    with img_col2:
        st.image(
            result["suspected"],
            caption=t["suspected_image"]
        )

    with img_col3:
        st.image(
            result["localization"],
            caption=t["tamper_localization"]
        )

    st.subheader(t["diff_heatmap"])

    map_col1, map_col2, map_col3 = st.columns(3)

    with map_col1:
        st.image(
            result["diff"],
            caption=t["difference_map"],
            clamp=True
        )

    with map_col2:
        st.image(
            result["threshold"],
            caption=t["threshold_map"],
            clamp=True
        )

    with map_col3:
        st.image(
            result["heatmap"],
            caption=t["tamper_heatmap"]
        )

    st.subheader(t["region_stats"])

    if result["region_data"]:
        region_df = pd.DataFrame(result["region_data"])
        st.dataframe(
            region_df,
            use_container_width=True
        )
    else:
        region_df = pd.DataFrame()
        st.info(t["no_regions"])

    # =========================
    # REPORT EXPORT
    # =========================

    st.markdown("---")
    st.subheader(t["report_section"])

    report_col1, report_col2 = st.columns(2)

    html_report = generate_html_report(
        batch_df=batch_df,
        selected_name=selected_image_name,
        result=result,
        region_df=region_df
    )

    csv_report = dataframe_to_csv_bytes(batch_df)

    with report_col1:
        st.download_button(
            label=t["html_report"],
            data=html_report,
            file_name="forensics_report.html",
            mime="text/html"
        )

    with report_col2:
        st.download_button(
            label=t["csv_report"],
            data=csv_report,
            file_name="forensics_batch_results.csv",
            mime="text/csv"
        )

else:
    st.info(t["info_upload"])
