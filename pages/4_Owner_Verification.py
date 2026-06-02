import streamlit as st
import cv2
import numpy as np
import pandas as pd
import html
from PIL import Image

st.set_page_config(
    page_title="Owner Verification",
    layout="wide"
)

# =========================
# LANGUAGE
# =========================

language = st.sidebar.selectbox(
    "🌐 Language / Dil",
    ["🇦🇿 Azərbaycan", "🇬🇧 English"]
)

lang = "az" if language == "🇦🇿 Azərbaycan" else "en"

T = {
    "az": {
        "title": "🛡️ Su nişanı əsasında müəlliflik yoxlaması",
        "desc": "Watermarked şəkli və bir neçə namizəd watermark yükləyin. Sistem ən uyğun watermark sahibini müəyyən etməyə çalışacaq.",
        "upload_image": "Watermarked şəkil yüklə",
        "upload_candidates": "Namizəd watermark-ları yüklə",
        "analysis_mode": "Analiz rejimi",
        "frequency": "Tezlik əsaslı blind matching",
        "visual": "Vizual watermark matching",
        "result": "Nəticə",
        "best_match": "Ən uyğun watermark",
        "confidence": "Uyğunluq göstəricisi",
        "status": "Status",
        "verified": "Mümkün sahib tapıldı",
        "uncertain": "Nəticə qeyri-müəyyəndir",
        "no_match": "Uyğun sahib tapılmadı",
        "table": "Namizəd watermark müqayisəsi",
        "image": "Watermarked şəkil",
        "candidate": "Namizəd watermark",
        "score": "Score",
        "correlation": "Correlation",
        "ber": "BER",
        "download_html": "HTML hesabatı yüklə",
        "download_csv": "CSV nəticələrini yüklə",
        "info": "Başlamaq üçün watermarked şəkil və ən azı iki namizəd watermark yükləyin."
    },
    "en": {
        "title": "🛡️ Watermark-Based Ownership Verification",
        "desc": "Upload a watermarked image and multiple candidate watermarks. The system will estimate the most likely owner.",
        "upload_image": "Upload watermarked image",
        "upload_candidates": "Upload candidate watermarks",
        "analysis_mode": "Analysis mode",
        "frequency": "Frequency-based blind matching",
        "visual": "Visual watermark matching",
        "result": "Result",
        "best_match": "Best matching watermark",
        "confidence": "Matching confidence",
        "status": "Status",
        "verified": "Likely owner found",
        "uncertain": "Uncertain result",
        "no_match": "No reliable owner found",
        "table": "Candidate watermark comparison",
        "image": "Watermarked image",
        "candidate": "Candidate watermark",
        "score": "Score",
        "correlation": "Correlation",
        "ber": "BER",
        "download_html": "Download HTML Report",
        "download_csv": "Download CSV Results",
        "info": "Please upload a watermarked image and at least two candidate watermarks."
    }
}

t = T[lang]

st.title(t["title"])
st.write(t["desc"])


# =========================
# HELPERS
# =========================

def preprocess_image(file, size=(512, 512)):
    img = Image.open(file).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, gray


def preprocess_watermark(file, size=(32, 32)):
    wm = Image.open(file).convert("L")
    wm = np.array(wm)
    wm = cv2.resize(wm, size)
    _, binary = cv2.threshold(wm, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.uint8)


def calculate_correlation(a, b):
    a = a.flatten()
    b = b.flatten()

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def calculate_ber(a, b):
    return float(np.sum(a != b) / a.size)


def normalize_score(corr, ber):
    corr_norm = max(0, corr)
    ber_quality = 1 - ber
    return round((0.65 * corr_norm + 0.35 * ber_quality) * 100, 2)


def extract_frequency_signature(image_gray, wm_shape=(32, 32)):
    """
    Blind heuristic signature extraction.
    This does not recover the exact watermark.
    It estimates a watermark-like binary signature from DCT block statistics.
    """
    image_gray = cv2.resize(image_gray, (256, 256))
    image_float = np.float32(image_gray)

    h, w = wm_shape
    block_size = 8
    signature = np.zeros((h, w), dtype=np.uint8)

    coeffs = [(3, 3), (3, 4), (4, 3), (4, 4)]

    for i in range(h):
        for j in range(w):
            x = i * block_size
            y = j * block_size

            block = image_float[x:x + block_size, y:y + block_size]
            dct_block = cv2.dct(block)

            vals = [dct_block[c] for c in coeffs]
            signature[i, j] = 1 if np.mean(vals) > 0 else 0

    return signature


def visual_signature_from_image(image_gray, wm_shape=(32, 32)):
    small = cv2.resize(image_gray, (wm_shape[1], wm_shape[0]))
    _, binary = cv2.threshold(
        small,
        0,
        1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary.astype(np.uint8)


def get_status(score):
    if score >= 75:
        return t["verified"]
    elif score >= 55:
        return t["uncertain"]
    else:
        return t["no_match"]


def generate_html_report(df, best_name, best_score, status):
    table_html = df.to_html(index=False, escape=True)

    report = f"""
    <!DOCTYPE html>
    <html lang="{lang}">
    <head>
        <meta charset="UTF-8">
        <title>{html.escape(t['title'])}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 35px;
                color: #222;
            }}
            h1, h2 {{
                color: #1f4e79;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
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
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background: #fafafa;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{html.escape(t['title'])}</h1>

        <div class="box">
            <h2>{html.escape(t['result'])}</h2>
            <p><b>{html.escape(t['best_match'])}:</b> {html.escape(best_name)}</p>
            <p><b>{html.escape(t['confidence'])}:</b> {best_score:.2f}%</p>
            <p><b>{html.escape(t['status'])}:</b> {html.escape(status)}</p>
        </div>

        <h2>{html.escape(t['table'])}</h2>
        {table_html}
    </body>
    </html>
    """

    return report.encode("utf-8")


# =========================
# UI
# =========================

analysis_mode = st.sidebar.radio(
    t["analysis_mode"],
    [
        t["frequency"],
        t["visual"]
    ]
)

watermarked_file = st.file_uploader(
    t["upload_image"],
    type=["png", "jpg", "jpeg"],
    key="watermarked_image"
)

candidate_files = st.file_uploader(
    t["upload_candidates"],
    type=["png", "jpg", "jpeg"],
    key="candidate_watermarks",
    accept_multiple_files=True
)


# =========================
# MAIN
# =========================

if watermarked_file and candidate_files and len(candidate_files) >= 2:

    img_rgb, img_gray = preprocess_image(watermarked_file)

    if analysis_mode == t["frequency"]:
        estimated_signature = extract_frequency_signature(img_gray)
    else:
        estimated_signature = visual_signature_from_image(img_gray)

    results = []
    candidate_previews = []

    for file in candidate_files:
        candidate_wm = preprocess_watermark(file)

        corr = calculate_correlation(estimated_signature, candidate_wm)
        ber = calculate_ber(estimated_signature, candidate_wm)
        score = normalize_score(corr, ber)

        results.append({
            t["candidate"]: file.name,
            t["correlation"]: round(corr, 4),
            t["ber"]: round(ber, 4),
            t["score"]: score
        })

        candidate_previews.append({
            "name": file.name,
            "watermark": candidate_wm
        })

    df = pd.DataFrame(results)
    best_row = df.loc[df[t["score"]].idxmax()]

    best_name = best_row[t["candidate"]]
    best_score = float(best_row[t["score"]])
    status = get_status(best_score)

    st.subheader(t["result"])

    c1, c2, c3 = st.columns(3)

    c1.metric(t["best_match"], best_name)
    c2.metric(t["confidence"], f"{best_score:.2f}%")
    c3.metric(t["status"], status)

    if best_score >= 75:
        st.success(status)
    elif best_score >= 55:
        st.warning(status)
    else:
        st.error(status)

    st.subheader(t["image"])

    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.image(
            img_rgb,
            caption=t["image"]
        )

    with img_col2:
        st.image(
            estimated_signature * 255,
            caption="Estimated Blind Signature",
            clamp=True
        )

    st.subheader(t["table"])

    st.dataframe(
        df.sort_values(by=t["score"], ascending=False),
        use_container_width=True
    )

    st.subheader("Candidate Watermark Preview")

    cols = st.columns(min(len(candidate_previews), 4))

    for idx, item in enumerate(candidate_previews):
        with cols[idx % len(cols)]:
            st.image(
                item["watermark"] * 255,
                caption=item["name"],
                clamp=True
            )

    st.markdown("---")

    html_report = generate_html_report(
        df=df.sort_values(by=t["score"], ascending=False),
        best_name=best_name,
        best_score=best_score,
        status=status
    )

    csv_report = df.to_csv(index=False).encode("utf-8-sig")

    r1, r2 = st.columns(2)

    with r1:
        st.download_button(
            label=t["download_html"],
            data=html_report,
            file_name="ownership_verification_report.html",
            mime="text/html"
        )

    with r2:
        st.download_button(
            label=t["download_csv"],
            data=csv_report,
            file_name="ownership_verification_results.csv",
            mime="text/csv"
        )

else:
    st.info(t["info"])
