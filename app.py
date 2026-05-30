import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(
    page_title="Context-Aware Adaptive Watermarking",
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
        "title": "Kontekst əsaslı adaptiv görünməz su nişanlama sistemi",
        "desc": "Şəkil və su nişanı yükləyin, domeni seçin və sistem uyğun alpha dəyərini təyin etsin.",
        "settings": "Parametrlər",
        "domain": "Şəkil domenini seçin",
        "attack": "Hücum növünü seçin",
        "jpeg_quality": "JPEG keyfiyyəti",
        "noise_strength": "Səs-küy intensivliyi",
        "blur_kernel": "Bulanıqlıq nüvəsi",
        "upload_host": "Host şəkli yüklə",
        "upload_wm": "Su nişanı yüklə (istəyə bağlı)",
        "adaptive_decision": "Adaptiv qərar",
        "selected_domain": "Seçilmiş domen",
        "predicted_alpha": "Təyin edilmiş alpha",
        "method": "Embedding metodu",
        "images": "Şəkillər",
        "original": "Orijinal şəkil",
        "input_wm": "Daxil edilən su nişanı",
        "watermarked": "Su nişanı yerləşdirilmiş şəkil",
        "after_attack": "Hücumdan sonra",
        "extracted": "Çıxarılmış su nişanı",
        "result_table": "Nəticə cədvəli",
        "metric": "Metrik",
        "value": "Dəyər",
        "info": "Başlamaq üçün host şəkli yükləyin.",
        "metric_exp": "Metriklərin izahı",
        "psnr_exp": "PSNR — orijinal və su nişanlı şəkil arasındakı keyfiyyət fərqini ölçür. Yüksək olması daha yaxşıdır.",
        "ssim_exp": "SSIM — struktur oxşarlığını ölçür. 1-ə yaxın olması daha yaxşıdır.",
        "ber_exp": "BER — çıxarılan su nişanında bit səhvlərinin nisbətini göstərir. Kiçik olması daha yaxşıdır.",
        "corr_exp": "Correlation — orijinal və çıxarılan su nişanının oxşarlığını ölçür. 1-ə yaxın olması daha yaxşıdır.",
    },
    "en": {
        "title": "Context-Aware Adaptive Invisible Watermarking System",
        "desc": "Upload a host image and watermark, choose the image domain, and the system will select an adaptive alpha value.",
        "settings": "Settings",
        "domain": "Select image domain",
        "attack": "Select attack",
        "jpeg_quality": "JPEG Quality",
        "noise_strength": "Noise Strength",
        "blur_kernel": "Blur Kernel",
        "upload_host": "Upload host image",
        "upload_wm": "Upload watermark image (optional)",
        "adaptive_decision": "Adaptive Decision",
        "selected_domain": "Selected domain",
        "predicted_alpha": "Predicted alpha",
        "method": "Embedding method",
        "images": "Images",
        "original": "Original Image",
        "input_wm": "Input Watermark",
        "watermarked": "Watermarked Image",
        "after_attack": "After Attack",
        "extracted": "Extracted Watermark",
        "result_table": "Result Table",
        "metric": "Metric",
        "value": "Value",
        "info": "Please upload a host image to start.",
        "metric_exp": "Metric Explanations",
        "psnr_exp": "PSNR measures the quality difference between the original and watermarked image. Higher is better.",
        "ssim_exp": "SSIM measures structural similarity. Values closer to 1 are better.",
        "ber_exp": "BER measures the bit error rate of the extracted watermark. Lower is better.",
        "corr_exp": "Correlation measures similarity between the original and extracted watermark. Values closer to 1 are better.",
    }
}

t = T[lang]

domain_options = {
    "az": ["Tibbi", "Peyk / GIS", "Mədəni irs", "Təbii"],
    "en": ["Medical", "Satellite / GIS", "Cultural Heritage", "Natural"]
}

attack_options = {
    "az": ["Hücum yoxdur", "JPEG sıxılma", "Gaussian səs-küy", "Gaussian bulanıqlıq"],
    "en": ["No Attack", "JPEG Compression", "Gaussian Noise", "Gaussian Blur"]
}

st.title(t["title"])
st.write(t["desc"])

# =========================
# DEFAULT WATERMARK
# =========================

def create_default_watermark():
    watermark = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(watermark, "W", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1, 2)
    return watermark

# =========================
# METRICS
# =========================

def calculate_psnr(original, processed):
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(original, processed):
    return ssim(original, processed, data_range=255)

def calculate_ber(original_wm, extracted_wm):
    return np.sum(original_wm != extracted_wm) / original_wm.size

def calculate_correlation(original_wm, extracted_wm):
    o = original_wm.flatten()
    e = extracted_wm.flatten()
    if np.std(o) == 0 or np.std(e) == 0:
        return 0
    return np.corrcoef(o, e)[0, 1]

# =========================
# ATTACKS
# =========================

def normalize_attack_name(attack):
    mapping = {
        "Hücum yoxdur": "No Attack",
        "JPEG sıxılma": "JPEG Compression",
        "Gaussian səs-küy": "Gaussian Noise",
        "Gaussian bulanıqlıq": "Gaussian Blur",
    }
    return mapping.get(attack, attack)

def apply_attack(image, attack_type, param):
    attack_type = normalize_attack_name(attack_type)
    attacked = image.copy()

    if attack_type == "JPEG Compression":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(param)]
        _, encimg = cv2.imencode(".jpg", attacked, encode_param)
        attacked = cv2.imdecode(encimg, 0)

    elif attack_type == "Gaussian Noise":
        noise = np.random.normal(0, param * 255, attacked.shape)
        attacked = attacked + noise
        attacked = np.clip(attacked, 0, 255).astype(np.uint8)

    elif attack_type == "Gaussian Blur":
        k = int(param)
        if k % 2 == 0:
            k += 1
        attacked = cv2.GaussianBlur(attacked, (k, k), 0)

    return attacked

# =========================
# BLOCK-SVD WATERMARKING
# =========================

def embed_watermark_svd_block(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)
    watermarked = host_float.copy()
    wm_h, wm_w = watermark_binary.shape
    block_size = 8

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size
            block = watermarked[x:x+block_size, y:y+block_size]

            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            if watermark_binary[i, j] == 1:
                S[0] += alpha
            else:
                S[0] -= alpha

            modified_block = np.dot(U, np.dot(np.diag(S), Vt))
            watermarked[x:x+block_size, y:y+block_size] = modified_block

    return np.uint8(np.clip(watermarked, 0, 255))

def extract_watermark_svd_block(original_image, watermarked_image, watermark_shape):
    original_float = np.float32(original_image)
    watermarked_float = np.float32(watermarked_image)
    wm_h, wm_w = watermark_shape
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)
    block_size = 8

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            original_block = original_float[x:x+block_size, y:y+block_size]
            watermarked_block = watermarked_float[x:x+block_size, y:y+block_size]

            _, S_original, _ = np.linalg.svd(original_block, full_matrices=False)
            _, S_watermarked, _ = np.linalg.svd(watermarked_block, full_matrices=False)

            extracted[i, j] = 1 if S_watermarked[0] - S_original[0] > 0 else 0

    return extracted

# =========================
# CONTEXT-AWARE ALPHA RULE
# =========================

def predict_alpha_by_domain(domain):
    if domain in ["Medical", "Tibbi"]:
        return 10
    elif domain in ["Cultural Heritage", "Mədəni irs"]:
        return 10
    elif domain in ["Satellite / GIS", "Peyk / GIS"]:
        return 20
    elif domain in ["Natural", "Təbii"]:
        return 10
    return 10

# =========================
# SIDEBAR SETTINGS
# =========================

st.sidebar.header(t["settings"])

domain = st.sidebar.selectbox(t["domain"], domain_options[lang])
attack_type = st.sidebar.selectbox(t["attack"], attack_options[lang])

attack_type_normalized = normalize_attack_name(attack_type)

if attack_type_normalized == "JPEG Compression":
    attack_param = st.sidebar.slider(t["jpeg_quality"], 10, 100, 70)
elif attack_type_normalized == "Gaussian Noise":
    attack_param = st.sidebar.slider(t["noise_strength"], 0.01, 0.10, 0.03)
elif attack_type_normalized == "Gaussian Blur":
    attack_param = st.sidebar.slider(t["blur_kernel"], 3, 9, 5, step=2)
else:
    attack_param = 0

uploaded_file = st.file_uploader(t["upload_host"], type=["png", "jpg", "jpeg"], key="host_upload")
watermark_file = st.file_uploader(t["upload_wm"], type=["png", "jpg", "jpeg"], key="watermark_upload")

# =========================
# METRIC EXPLANATIONS
# =========================

with st.expander(t["metric_exp"]):
    st.write(f"**PSNR:** {t['psnr_exp']}")
    st.write(f"**SSIM:** {t['ssim_exp']}")
    st.write(f"**BER:** {t['ber_exp']}")
    st.write(f"**Correlation:** {t['corr_exp']}")

# =========================
# MAIN PROCESS
# =========================

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_rgb = cv2.resize(img_rgb, (512, 512))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if watermark_file is not None:
        wm_img = Image.open(watermark_file).convert("L")
        wm_img = np.array(wm_img)
        wm_img = cv2.resize(wm_img, (32, 32))
        _, watermark_binary = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)
    else:
        watermark_binary = create_default_watermark()

    predicted_alpha = predict_alpha_by_domain(domain)

    watermarked = embed_watermark_svd_block(img_gray, watermark_binary, alpha=predicted_alpha)
    attacked = apply_attack(watermarked, attack_type, attack_param)

    extracted = extract_watermark_svd_block(img_gray, attacked, watermark_binary.shape)

    psnr_val = calculate_psnr(img_gray, watermarked)
    ssim_val = calculate_ssim(img_gray, watermarked)
    ber_val = calculate_ber(watermark_binary, extracted)
    corr_val = calculate_correlation(watermark_binary, extracted)

    st.subheader(t["adaptive_decision"])
    st.write(f"{t['selected_domain']}: **{domain}**")
    st.write(f"{t['predicted_alpha']}: **{predicted_alpha}**")
    st.write(f"{t['method']}: **Block-SVD**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PSNR", f"{psnr_val:.2f} dB")
    col2.metric("SSIM", f"{ssim_val:.4f}")
    col3.metric("BER", f"{ber_val:.4f}")
    col4.metric("Correlation", f"{corr_val:.4f}")

    st.subheader(t["images"])

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.image(img_gray, caption=t["original"], clamp=True)
    with c2:
        st.image(watermark_binary * 255, caption=t["input_wm"], clamp=True)
    with c3:
        st.image(watermarked, caption=f"{t['watermarked']} α={predicted_alpha}", clamp=True)
    with c4:
        st.image(attacked, caption=f"{t['after_attack']}: {attack_type}", clamp=True)
    with c5:
        st.image(extracted * 255, caption=t["extracted"], clamp=True)

    st.subheader(t["result_table"])

    result_df = pd.DataFrame({
        t["metric"]: ["Domain", "Method", "Alpha", "Attack", "PSNR", "SSIM", "BER", "Correlation"],
        t["value"]: [
            domain,
            "Block-SVD",
            predicted_alpha,
            attack_type,
            round(psnr_val, 4),
            round(ssim_val, 4),
            round(ber_val, 4),
            round(corr_val, 4)
        ]
    })

    st.dataframe(result_df, use_container_width=True)

else:
    st.info(t["info"])
