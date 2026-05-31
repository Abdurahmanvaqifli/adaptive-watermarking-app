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
        "upload_wm": "Su nişanı yüklə",
        "adaptive_alpha": "Adaptiv Alpha",
        "alpha_selection": "Alpha seçimi",
        "recommended": "Tövsiyə olunan",
        "manual": "Əl ilə seçim",
        "alpha_value": "Alpha dəyəri",
        "embedding_method": "Embedding metodu",
        "adaptive_decision": "Adaptiv qərar",
        "selected_domain": "Seçilmiş domen",
        "recommended_alpha": "Tövsiyə olunan alpha",
        "selected_alpha": "İstifadə olunan alpha",
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
        "watermark_type": "Su nişanı növü",
        "default_watermark": "Standart su nişanı",
        "upload_logo": "Logo yüklə",
        "text_watermark": "Mətn su nişanı",
        "enter_watermark_text": "Su nişanı mətnini daxil edin",
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
        "upload_wm": "Upload watermark image",
        "adaptive_alpha": "Adaptive Alpha",
        "alpha_selection": "Alpha selection",
        "recommended": "Recommended",
        "manual": "Manual",
        "alpha_value": "Alpha value",
        "embedding_method": "Embedding method",
        "adaptive_decision": "Adaptive Decision",
        "selected_domain": "Selected domain",
        "recommended_alpha": "Recommended alpha",
        "selected_alpha": "Selected alpha",
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
        "watermark_type": "Watermark type",
        "default_watermark": "Default watermark",
        "upload_logo": "Upload logo",
        "text_watermark": "Text watermark",
        "enter_watermark_text": "Enter watermark text",
    }
}

t = T[lang]

domain_options = {
    "az": ["Tibbi", "Peyk / GIS", "Mədəni irs", "Təbii"],
    "en": ["Medical", "Satellite / GIS", "Cultural Heritage", "Natural"]
}

attack_options = {
    "az": [
        "Hücum yoxdur",
        "JPEG sıxılma",
        "Gaussian səs-küy",
        "Salt & Pepper səs-küy",
        "Gaussian bulanıqlıq",
        "Parlaqlıq dəyişikliyi",
        "Kontrast dəyişikliyi",
        "Fırlatma",
        "Kəsmə"
    ],
    "en": [
        "No Attack",
        "JPEG Compression",
        "Gaussian Noise",
        "Salt & Pepper Noise",
        "Gaussian Blur",
        "Brightness Change",
        "Contrast Change",
        "Rotation",
        "Cropping"
    ]
}

st.title(t["title"])
st.write(t["desc"])

# =========================
# WATERMARK CREATION
# =========================

def create_default_watermark():
    watermark = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(watermark, "W", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1, 2)
    return watermark


def create_text_watermark(text):
    watermark = np.zeros((32, 32), dtype=np.uint8)

    text = text.strip()
    if text == "":
        text = "W"

    cv2.putText(
        watermark,
        text[:2],
        (3, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        1,
        1
    )

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
        "Salt & Pepper səs-küy": "Salt & Pepper Noise",
        "Gaussian bulanıqlıq": "Gaussian Blur",
        "Parlaqlıq dəyişikliyi": "Brightness Change",
        "Kontrast dəyişikliyi": "Contrast Change",
        "Fırlatma": "Rotation",
        "Kəsmə": "Cropping",
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

    elif attack_type == "Salt & Pepper Noise":
        amount = float(param)
        noisy = attacked.copy()

        num_salt = int(amount * attacked.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in attacked.shape]
        noisy[coords[0], coords[1]] = 255

        num_pepper = int(amount * attacked.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in attacked.shape]
        noisy[coords[0], coords[1]] = 0

        attacked = noisy

    elif attack_type == "Gaussian Blur":
        k = int(param)
        if k % 2 == 0:
            k += 1
        attacked = cv2.GaussianBlur(attacked, (k, k), 0)

    elif attack_type == "Brightness Change":
        attacked = cv2.convertScaleAbs(attacked, alpha=1.0, beta=int(param))

    elif attack_type == "Contrast Change":
        attacked = cv2.convertScaleAbs(attacked, alpha=float(param), beta=0)

    elif attack_type == "Rotation":
        h, w = attacked.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, float(param), 1.0)

        attacked = cv2.warpAffine(
            attacked,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

    elif attack_type == "Cropping":
        crop_percent = float(param)
        h, w = attacked.shape

        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)

        cropped = attacked[crop_h:h-crop_h, crop_w:w-crop_w]
        attacked = cv2.resize(cropped, (w, h))

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
# IMPROVED DCT WATERMARKING
# =========================

def embed_watermark_dct_multi(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)
    watermarked = host_float.copy()

    wm_h, wm_w = watermark_binary.shape
    block_size = 8

    coeffs = [
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4)
    ]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block = watermarked[x:x+block_size, y:y+block_size]
            dct_block = cv2.dct(block)

            bit = watermark_binary[i, j]

            for c in coeffs:
                if bit == 1:
                    dct_block[c] += alpha
                else:
                    dct_block[c] -= alpha

            idct_block = cv2.idct(dct_block)
            watermarked[x:x+block_size, y:y+block_size] = idct_block

    return np.uint8(np.clip(watermarked, 0, 255))


def extract_watermark_dct_multi(original_image, watermarked_image, watermark_shape):
    original_float = np.float32(original_image)
    watermarked_float = np.float32(watermarked_image)

    wm_h, wm_w = watermark_shape
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)

    block_size = 8

    coeffs = [
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4)
    ]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            original_block = original_float[x:x+block_size, y:y+block_size]
            watermarked_block = watermarked_float[x:x+block_size, y:y+block_size]

            dct_original = cv2.dct(original_block)
            dct_watermarked = cv2.dct(watermarked_block)

            diffs = []

            for c in coeffs:
                diffs.append(dct_watermarked[c] - dct_original[c])

            avg_diff = np.mean(diffs)

            if avg_diff > 0:
                extracted[i, j] = 1
            else:
                extracted[i, j] = 0

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
    else:
        return 10

# =========================
# SIDEBAR SETTINGS
# =========================

st.sidebar.header(t["settings"])

domain = st.sidebar.selectbox(t["domain"], domain_options[lang])
attack_type = st.sidebar.selectbox(t["attack"], attack_options[lang])

method_options = [
    "Block-SVD",
    "Improved DCT"
]

selected_method = st.sidebar.selectbox(
    t["embedding_method"],
    method_options
)

recommended_alpha = predict_alpha_by_domain(domain)

st.sidebar.markdown("---")
st.sidebar.write(f"### {t['adaptive_alpha']}")

alpha_mode = st.sidebar.radio(
    t["alpha_selection"],
    [t["recommended"], t["manual"]]
)

if alpha_mode == t["recommended"]:
    predicted_alpha = recommended_alpha
else:
    predicted_alpha = st.sidebar.slider(
        t["alpha_value"],
        5,
        50,
        recommended_alpha,
        step=5
    )

attack_type_normalized = normalize_attack_name(attack_type)

if attack_type_normalized == "JPEG Compression":
    attack_param = st.sidebar.slider(t["jpeg_quality"], 10, 100, 70)

elif attack_type_normalized == "Gaussian Noise":
    attack_param = st.sidebar.slider(t["noise_strength"], 0.01, 0.10, 0.03)

elif attack_type_normalized == "Salt & Pepper Noise":
    label = "Salt & Pepper intensivliyi" if lang == "az" else "Salt & Pepper Strength"
    attack_param = st.sidebar.slider(label, 0.01, 0.10, 0.03)

elif attack_type_normalized == "Gaussian Blur":
    attack_param = st.sidebar.slider(t["blur_kernel"], 3, 9, 5, step=2)

elif attack_type_normalized == "Brightness Change":
    label = "Parlaqlıq səviyyəsi" if lang == "az" else "Brightness Level"
    attack_param = st.sidebar.slider(label, -80, 80, 30)

elif attack_type_normalized == "Contrast Change":
    label = "Kontrast səviyyəsi" if lang == "az" else "Contrast Level"
    attack_param = st.sidebar.slider(label, 0.5, 2.0, 1.3)

elif attack_type_normalized == "Rotation":
    label = "Fırlatma bucağı" if lang == "az" else "Rotation Angle"
    attack_param = st.sidebar.slider(label, -30, 30, 10)

elif attack_type_normalized == "Cropping":
    label = "Kəsmə faizi" if lang == "az" else "Cropping Percentage"
    attack_param = st.sidebar.slider(label, 0.05, 0.30, 0.10)

else:
    attack_param = 0

uploaded_file = st.file_uploader(
    t["upload_host"],
    type=["png", "jpg", "jpeg"],
    key="host_upload"
)

watermark_type = st.sidebar.radio(
    t["watermark_type"],
    [
        t["default_watermark"],
        t["upload_logo"],
        t["text_watermark"]
    ]
)

watermark_file = None
watermark_text = ""

if watermark_type == t["upload_logo"]:
    watermark_file = st.file_uploader(
        t["upload_wm"],
        type=["png", "jpg", "jpeg"],
        key="watermark_upload"
    )

elif watermark_type == t["text_watermark"]:
    watermark_text = st.sidebar.text_input(
        t["enter_watermark_text"],
        value="AV"
    )

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

    if watermark_type == t["upload_logo"] and watermark_file is not None:
        wm_img = Image.open(watermark_file).convert("L")
        wm_img = np.array(wm_img)
        wm_img = cv2.resize(wm_img, (32, 32))

        _, watermark_binary = cv2.threshold(
            wm_img,
            127,
            1,
            cv2.THRESH_BINARY
        )

    elif watermark_type == t["text_watermark"]:
        watermark_binary = create_text_watermark(watermark_text)

    else:
        watermark_binary = create_default_watermark()

    if selected_method == "Block-SVD":
        watermarked = embed_watermark_svd_block(
            img_gray,
            watermark_binary,
            alpha=predicted_alpha
        )

        attacked = apply_attack(
            watermarked,
            attack_type,
            attack_param
        )

        extracted = extract_watermark_svd_block(
            img_gray,
            attacked,
            watermark_binary.shape
        )

    elif selected_method == "Improved DCT":
        watermarked = embed_watermark_dct_multi(
            img_gray,
            watermark_binary,
            alpha=predicted_alpha
        )

        attacked = apply_attack(
            watermarked,
            attack_type,
            attack_param
        )

        extracted = extract_watermark_dct_multi(
            img_gray,
            attacked,
            watermark_binary.shape
        )

    else:
        st.error("Unknown embedding method selected.")
        st.stop()

    psnr_val = calculate_psnr(img_gray, watermarked)
    ssim_val = calculate_ssim(img_gray, watermarked)
    ber_val = calculate_ber(watermark_binary, extracted)
    corr_val = calculate_correlation(watermark_binary, extracted)

    st.subheader(t["adaptive_decision"])
    st.write(f"{t['selected_domain']}: **{domain}**")
    st.write(f"{t['recommended_alpha']}: **{recommended_alpha}**")
    st.write(f"{t['selected_alpha']}: **{predicted_alpha}**")
    st.write(f"{t['method']}: **{selected_method}**")

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
        t["metric"]: [
            "Domain",
            "Method",
            "Recommended Alpha",
            "Selected Alpha",
            "Attack",
            "PSNR",
            "SSIM",
            "BER",
            "Correlation"
        ],
        t["value"]: [
            domain,
            selected_method,
            recommended_alpha,
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
