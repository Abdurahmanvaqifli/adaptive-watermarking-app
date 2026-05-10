
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

st.title("Context-Aware Adaptive Invisible Watermarking System")
st.write("Upload a host image and watermark, choose the image domain, and the system will select an adaptive alpha value.")

# =========================
# DEFAULT WATERMARK
# =========================

def create_default_watermark():
    watermark = np.zeros((32, 32), dtype=np.uint8)
    cv2.putText(
        watermark,
        "W",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        1,
        2
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

def apply_attack(image, attack_type, param):
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

    elif attack_type == "No Attack":
        attacked = image.copy()

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

    watermarked = np.clip(watermarked, 0, 255)
    return np.uint8(watermarked)

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

            diff = S_watermarked[0] - S_original[0]

            if diff > 0:
                extracted[i, j] = 1
            else:
                extracted[i, j] = 0

    return extracted

# =========================
# CONTEXT-AWARE ALPHA RULE
# =========================

def predict_alpha_by_domain(domain):
    if domain == "Medical":
        return 10
    elif domain == "Cultural Heritage":
        return 10
    elif domain == "Satellite / GIS":
        return 20
    elif domain == "Natural":
        return 10
    else:
        return 10

# =========================
# SIDEBAR SETTINGS
# =========================

st.sidebar.header("Settings")

domain = st.sidebar.selectbox(
    "Select image domain",
    ["Medical", "Satellite / GIS", "Cultural Heritage", "Natural"]
)

attack_type = st.sidebar.selectbox(
    "Select attack",
    ["No Attack", "JPEG Compression", "Gaussian Noise", "Gaussian Blur"]
)

if attack_type == "JPEG Compression":
    attack_param = st.sidebar.slider("JPEG Quality", 10, 100, 70)
elif attack_type == "Gaussian Noise":
    attack_param = st.sidebar.slider("Noise Strength", 0.01, 0.10, 0.03)
elif attack_type == "Gaussian Blur":
    attack_param = st.sidebar.slider("Blur Kernel", 3, 9, 5, step=2)
else:
    attack_param = 0

uploaded_file = st.file_uploader(
    "Upload host image",
    type=["png", "jpg", "jpeg"],
    key="host_upload"
)

watermark_file = st.file_uploader(
    "Upload watermark image (optional)",
    type=["png", "jpg", "jpeg"],
    key="watermark_upload"
)

# =========================
# MAIN PROCESS
# =========================

if uploaded_file is not None:

    # Host image
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_rgb = cv2.resize(img_rgb, (512, 512))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Watermark image
    if watermark_file is not None:
        wm_img = Image.open(watermark_file).convert("L")
        wm_img = np.array(wm_img)
        wm_img = cv2.resize(wm_img, (32, 32))

        _, watermark_binary = cv2.threshold(
            wm_img,
            127,
            1,
            cv2.THRESH_BINARY
        )
    else:
        watermark_binary = create_default_watermark()

    # Predict alpha
    predicted_alpha = predict_alpha_by_domain(domain)

    # Embed watermark
    watermarked = embed_watermark_svd_block(
        img_gray,
        watermark_binary,
        alpha=predicted_alpha
    )

    # Apply attack
    attacked = apply_attack(
        watermarked,
        attack_type,
        attack_param
    )

    # Extract watermark
    extracted = extract_watermark_svd_block(
        img_gray,
        attacked,
        watermark_binary.shape
    )

    # Metrics
    psnr_val = calculate_psnr(img_gray, watermarked)
    ssim_val = calculate_ssim(img_gray, watermarked)
    ber_val = calculate_ber(watermark_binary, extracted)
    corr_val = calculate_correlation(watermark_binary, extracted)

    st.subheader("Adaptive Decision")
    st.write(f"Selected domain: **{domain}**")
    st.write(f"Predicted alpha: **{predicted_alpha}**")
    st.write("Embedding method: **Block-SVD**")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PSNR", f"{psnr_val:.2f} dB")
    col2.metric("SSIM", f"{ssim_val:.4f}")
    col3.metric("BER", f"{ber_val:.4f}")
    col4.metric("Correlation", f"{corr_val:.4f}")

    st.subheader("Images")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.image(img_gray, caption="Original Image", clamp=True)

    with c2:
        st.image(watermark_binary * 255, caption="Input Watermark", clamp=True)

    with c3:
        st.image(watermarked, caption=f"Watermarked Image α={predicted_alpha}", clamp=True)

    with c4:
        st.image(attacked, caption=f"After Attack: {attack_type}", clamp=True)

    with c5:
        st.image(extracted * 255, caption="Extracted Watermark", clamp=True)

    st.subheader("Result Table")

    result_df = pd.DataFrame({
        "Metric": [
            "Domain",
            "Method",
            "Alpha",
            "Attack",
            "PSNR",
            "SSIM",
            "BER",
            "Correlation"
        ],
        "Value": [
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
    st.info("Please upload a host image to start.")
