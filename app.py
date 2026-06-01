import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pywt
import io
import zipfile
import re
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
        "result_table": "Seçilmiş metod üzrə nəticə cədvəli",
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
        "comparison_title": "Bütün embedding metodlarının müqayisəsi",
        "comparison_desc": "Bu müqayisə eyni alpha, eyni watermark və eyni attack parametrləri altında aparılır.",
        "imperceptibility": "Imperceptibility müqayisəsi",
        "robustness": "Robustness müqayisəsi",
        "psnr_chart": "PSNR müqayisəsi",
        "ssim_chart": "SSIM müqayisəsi",
        "ber_chart": "BER müqayisəsi",
        "corr_chart": "Correlation müqayisəsi",
        "score": "Score",
        "recommended_method": "Tövsiyə olunan metod",
        "recommendation_reason": "Tövsiyə səbəbi",
        "highest_score_reason": "Cari alpha, watermark və attack parametrləri altında ən yüksək score göstərən metod.",
        "max_images": "Maksimum şəkil sayı",
        "batch_summary": "Batch emal nəticələri",
        "batch_completed": "Batch emalı tamamlandı.",
        "batch_score_chart": "Batch Score qrafiki",
        "download_batch": "Batch nəticələrini yüklə",
        "download_watermarked_zip": "Watermarked şəkilləri ZIP yüklə",
        "download_attacked_zip": "Attacked şəkilləri ZIP yüklə",
        "download_extracted_zip": "Çıxarılmış watermark-ları ZIP yüklə",
        "score_comparison": "Score müqayisəsi",
        "visual_gallery": "Vizual çıxarış qalereyası",
        "embedding_channel": "Embedding kanalı",
        "grayscale": "Grayscale",
        "red_channel": "Qırmızı kanal",
        "green_channel": "Yaşıl kanal",
        "blue_channel": "Mavi kanal",
        "attack_mode": "Attack rejimi",
        "single_attack": "Tək attack",
        "combined_attack": "Kombinə attack",
        "combined_attack_type": "Kombinə attack növü",
        "jpeg_noise": "JPEG + Gaussian səs-küy",
        "jpeg_blur": "JPEG + Gaussian bulanıqlıq",
        "jpeg_noise_blur": "JPEG + Gaussian səs-küy + Gaussian bulanıqlıq",
        "jpeg_noise_rotation": "JPEG + Gaussian səs-küy + Fırlatma",
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
        "result_table": "Selected Method Result Table",
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
        "comparison_title": "Comparison of All Embedding Methods",
        "comparison_desc": "This comparison is performed under the same alpha, same watermark, and same attack settings.",
        "imperceptibility": "Imperceptibility Comparison",
        "robustness": "Robustness Comparison",
        "psnr_chart": "PSNR Comparison",
        "ssim_chart": "SSIM Comparison",
        "ber_chart": "BER Comparison",
        "corr_chart": "Correlation Comparison",
        "score": "Score",
        "recommended_method": "Recommended Method",
        "recommendation_reason": "Recommendation reason",
        "highest_score_reason": "The method with the highest score under the current alpha, watermark, and attack settings.",
        "max_images": "Maximum images",
        "batch_summary": "Batch Processing Summary",
        "batch_completed": "Batch processing completed.",
        "batch_score_chart": "Batch Score Chart",
        "download_batch": "Download Batch Outputs",
        "download_watermarked_zip": "Download Watermarked Images ZIP",
        "download_attacked_zip": "Download Attacked Images ZIP",
        "download_extracted_zip": "Download Extracted Watermarks ZIP",
        "score_comparison": "Score Comparison",
        "visual_gallery": "Visual Extraction Gallery",
        "embedding_channel": "Embedding channel",
        "grayscale": "Grayscale",
        "red_channel": "Red channel",
        "green_channel": "Green channel",
        "blue_channel": "Blue channel",
        "attack_mode": "Attack mode",
        "single_attack": "Single attack",
        "combined_attack": "Combined attack",
        "combined_attack_type": "Combined attack type",
        "jpeg_noise": "JPEG + Gaussian Noise",
        "jpeg_blur": "JPEG + Gaussian Blur",
        "jpeg_noise_blur": "JPEG + Gaussian Noise + Gaussian Blur",
        "jpeg_noise_rotation": "JPEG + Gaussian Noise + Rotation",
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
# DOMAIN-SPECIFIC SCORING
# =========================

def get_domain_weights(domain):
    if domain in ["Medical", "Tibbi"]:
        return {"psnr": 0.40, "ssim": 0.40, "corr": 0.10, "ber": 0.10}

    elif domain in ["Satellite / GIS", "Peyk / GIS"]:
        return {"psnr": 0.15, "ssim": 0.15, "corr": 0.35, "ber": 0.35}

    elif domain in ["Cultural Heritage", "Mədəni irs"]:
        return {"psnr": 0.35, "ssim": 0.35, "corr": 0.15, "ber": 0.15}

    else:
        return {"psnr": 0.25, "ssim": 0.25, "corr": 0.25, "ber": 0.25}


def calculate_score(psnr_val, ssim_val, ber_val, corr_val, domain):
    weights = get_domain_weights(domain)

    psnr_norm = min(psnr_val / 60.0, 1.0)
    ssim_norm = max(min(ssim_val, 1.0), 0.0)
    corr_norm = max(min(corr_val, 1.0), 0.0)
    ber_quality = 1.0 - max(min(ber_val, 1.0), 0.0)

    score = (
        weights["psnr"] * psnr_norm +
        weights["ssim"] * ssim_norm +
        weights["corr"] * corr_norm +
        weights["ber"] * ber_quality
    )

    return round(score, 4)

# =========================
# FILE / ZIP HELPERS
# =========================

def safe_filename(filename):
    filename = filename.rsplit(".", 1)[0]
    filename = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)
    return filename


def add_image_to_zip(zip_file, folder_name, file_name, image_array):
    image_array = np.uint8(np.clip(image_array, 0, 255))
    success, encoded_image = cv2.imencode(".png", image_array)

    if success:
        zip_file.writestr(
            f"{folder_name}/{file_name}.png",
            encoded_image.tobytes()
        )

# =========================
# RGB CHANNEL HELPERS
# =========================

def get_host_channel(img_rgb, channel_mode):
    if channel_mode == "Red":
        return img_rgb[:, :, 0]
    elif channel_mode == "Green":
        return img_rgb[:, :, 1]
    elif channel_mode == "Blue":
        return img_rgb[:, :, 2]
    else:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def reconstruct_display_image(original_rgb, processed_channel, channel_mode):
    if channel_mode == "Red":
        output = original_rgb.copy()
        output[:, :, 0] = processed_channel
        return output

    elif channel_mode == "Green":
        output = original_rgb.copy()
        output[:, :, 1] = processed_channel
        return output

    elif channel_mode == "Blue":
        output = original_rgb.copy()
        output[:, :, 2] = processed_channel
        return output

    else:
        return processed_channel


def get_channel_label(channel_mode):
    return channel_mode


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
# COMBINED ATTACKS
# =========================

def apply_combined_attack(image, attack_name):
    attacked = image.copy()

    if attack_name == "JPEG + Noise":
        attacked = apply_attack(attacked, "JPEG Compression", 60)
        attacked = apply_attack(attacked, "Gaussian Noise", 0.03)

    elif attack_name == "JPEG + Blur":
        attacked = apply_attack(attacked, "JPEG Compression", 60)
        attacked = apply_attack(attacked, "Gaussian Blur", 5)

    elif attack_name == "JPEG + Noise + Blur":
        attacked = apply_attack(attacked, "JPEG Compression", 60)
        attacked = apply_attack(attacked, "Gaussian Noise", 0.03)
        attacked = apply_attack(attacked, "Gaussian Blur", 5)

    elif attack_name == "JPEG + Noise + Rotation":
        attacked = apply_attack(attacked, "JPEG Compression", 60)
        attacked = apply_attack(attacked, "Gaussian Noise", 0.03)
        attacked = apply_attack(attacked, "Rotation", 10)

    return attacked


def apply_selected_attack(image, attack_mode, attack_type, attack_param, combined_attack):
    if attack_mode == "Combined Attack":
        return apply_combined_attack(image, combined_attack)

    return apply_attack(image, attack_type, attack_param)


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
    coeffs = [(3, 3), (3, 4), (4, 3), (4, 4)]

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
    coeffs = [(3, 3), (3, 4), (4, 3), (4, 4)]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            original_block = original_float[x:x+block_size, y:y+block_size]
            watermarked_block = watermarked_float[x:x+block_size, y:y+block_size]

            dct_original = cv2.dct(original_block)
            dct_watermarked = cv2.dct(watermarked_block)

            diffs = [dct_watermarked[c] - dct_original[c] for c in coeffs]
            avg_diff = np.mean(diffs)

            extracted[i, j] = 1 if avg_diff > 0 else 0

    return extracted


# =========================
# IMPROVED DWT WATERMARKING
# =========================

def embed_watermark_dwt_ll(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)
    LL, (LH, HL, HH) = pywt.dwt2(host_float, "haar")

    wm_resized = cv2.resize(
        watermark_binary.astype(np.float32),
        (LL.shape[1], LL.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    wm_signal = np.where(wm_resized > 0.5, 1, -1)
    LL_w = LL + alpha * wm_signal

    watermarked = pywt.idwt2(
        (LL_w, (LH, HL, HH)),
        "haar"
    )

    return np.uint8(np.clip(watermarked, 0, 255))


def extract_watermark_dwt_ll(original_image, watermarked_image, watermark_shape):
    original_float = np.float32(original_image)
    watermarked_float = np.float32(watermarked_image)

    LL_o, _ = pywt.dwt2(original_float, "haar")
    LL_w, _ = pywt.dwt2(watermarked_float, "haar")

    diff = LL_w - LL_o
    extracted_large = np.where(diff > 0, 1, 0).astype(np.uint8)

    extracted = cv2.resize(
        extracted_large,
        (watermark_shape[1], watermark_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return extracted


# =========================
# DCT-DWT WATERMARKING
# =========================

def embed_watermark_dct_dwt(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)
    LL, (LH, HL, HH) = pywt.dwt2(host_float, "haar")
    LL_w = LL.copy()

    wm_h, wm_w = watermark_binary.shape
    block_size = 8
    coeffs = [(3, 3), (4, 4)]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block = LL_w[x:x+block_size, y:y+block_size]
            dct_block = cv2.dct(block)
            bit = watermark_binary[i, j]

            for c in coeffs:
                if bit == 1:
                    dct_block[c] += alpha
                else:
                    dct_block[c] -= alpha

            LL_w[x:x+block_size, y:y+block_size] = cv2.idct(dct_block)

    watermarked = pywt.idwt2(
        (LL_w, (LH, HL, HH)),
        "haar"
    )

    return np.uint8(np.clip(watermarked, 0, 255))


def extract_watermark_dct_dwt(original_image, watermarked_image, watermark_shape):
    original_float = np.float32(original_image)
    watermarked_float = np.float32(watermarked_image)

    LL_o, _ = pywt.dwt2(original_float, "haar")
    LL_w, _ = pywt.dwt2(watermarked_float, "haar")

    wm_h, wm_w = watermark_shape
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)

    block_size = 8
    coeffs = [(3, 3), (4, 4)]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block_o = LL_o[x:x+block_size, y:y+block_size]
            block_w = LL_w[x:x+block_size, y:y+block_size]

            dct_o = cv2.dct(block_o)
            dct_w = cv2.dct(block_w)

            diffs = [dct_w[c] - dct_o[c] for c in coeffs]
            extracted[i, j] = 1 if np.mean(diffs) > 0 else 0

    return extracted


# =========================
# DWT-DFT WATERMARKING
# =========================

def embed_watermark_dwt_dft(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)

    LL, (LH, HL, HH) = pywt.dwt2(host_float, "haar")
    LL_w = LL.copy()

    wm_h, wm_w = watermark_binary.shape
    block_size = 8
    coeffs = [(3, 3), (3, 4), (4, 3), (4, 4)]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block = LL_w[x:x+block_size, y:y+block_size]
            dft_block = np.fft.fft2(block)

            bit = watermark_binary[i, j]

            for c in coeffs:
                if bit == 1:
                    dft_block[c] += alpha
                else:
                    dft_block[c] -= alpha

            LL_w[x:x+block_size, y:y+block_size] = np.real(np.fft.ifft2(dft_block))

    watermarked = pywt.idwt2(
        (LL_w, (LH, HL, HH)),
        "haar"
    )

    return np.uint8(np.clip(watermarked, 0, 255))


def extract_watermark_dwt_dft(original_image, watermarked_image, watermark_shape):
    original_float = np.float32(original_image)
    watermarked_float = np.float32(watermarked_image)

    LL_o, _ = pywt.dwt2(original_float, "haar")
    LL_w, _ = pywt.dwt2(watermarked_float, "haar")

    wm_h, wm_w = watermark_shape
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)

    block_size = 8
    coeffs = [(3, 3), (3, 4), (4, 3), (4, 4)]

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block_o = LL_o[x:x+block_size, y:y+block_size]
            block_w = LL_w[x:x+block_size, y:y+block_size]

            dft_o = np.fft.fft2(block_o)
            dft_w = np.fft.fft2(block_w)

            diffs = [np.real(dft_w[c] - dft_o[c]) for c in coeffs]
            extracted[i, j] = 1 if np.mean(diffs) > 0 else 0

    return extracted


# =========================
# DCT-SVD WATERMARKING
# =========================

def embed_watermark_dct_svd(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)

    dct_image = cv2.dct(host_float)
    dct_w = dct_image.copy()

    wm_h, wm_w = watermark_binary.shape
    block_size = 8

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block = dct_w[x:x+block_size, y:y+block_size]
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            if watermark_binary[i, j] == 1:
                S[0] += alpha
            else:
                S[0] -= alpha

            dct_w[x:x+block_size, y:y+block_size] = np.dot(
                U,
                np.dot(np.diag(S), Vt)
            )

    watermarked = cv2.idct(dct_w)

    return np.uint8(np.clip(watermarked, 0, 255))
    def extract_watermark_dct_svd(original_image, watermarked_image, watermark_shape):
        original_float = np.float32(original_image)
        watermarked_float = np.float32(watermarked_image)

    dct_o = cv2.dct(original_float)
    dct_w = cv2.dct(watermarked_float)

    wm_h, wm_w = watermark_shape
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)

    block_size = 8

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block_o = dct_o[x:x+block_size, y:y+block_size]
            block_w = dct_w[x:x+block_size, y:y+block_size]

            _, S_o, _ = np.linalg.svd(block_o, full_matrices=False)
            _, S_w, _ = np.linalg.svd(block_w, full_matrices=False)

            extracted[i, j] = 1 if S_w[0] - S_o[0] > 0 else 0

    return extracted


# =========================
# DWT-SVD WATERMARKING
# =========================

def embed_watermark_dwt_svd(host_gray, watermark_binary, alpha=10):
    host_float = np.float32(host_gray)

    LL, (LH, HL, HH) = pywt.dwt2(host_float, "haar")
    LL_w = LL.copy()

    wm_h, wm_w = watermark_binary.shape
    block_size = 8

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block = LL_w[x:x+block_size, y:y+block_size]
            U, S, Vt = np.linalg.svd(block, full_matrices=False)

            if watermark_binary[i, j] == 1:
                S[0] += alpha
            else:
                S[0] -= alpha

            LL_w[x:x+block_size, y:y+block_size] = np.dot(
                U,
                np.dot(np.diag(S), Vt)
            )

    watermarked = pywt.idwt2(
        (LL_w, (LH, HL, HH)),
        "haar"
    )

    return np.uint8(np.clip(watermarked, 0, 255))


def extract_watermark_dwt_svd(original_image, watermarked_image, watermark_shape):
    original_float = np.float32(original_image)
    watermarked_float = np.float32(watermarked_image)

    LL_o, _ = pywt.dwt2(original_float, "haar")
    LL_w, _ = pywt.dwt2(watermarked_float, "haar")

    wm_h, wm_w = watermark_shape
    extracted = np.zeros((wm_h, wm_w), dtype=np.uint8)

    block_size = 8

    for i in range(wm_h):
        for j in range(wm_w):
            x = i * block_size
            y = j * block_size

            block_o = LL_o[x:x+block_size, y:y+block_size]
            block_w = LL_w[x:x+block_size, y:y+block_size]

            _, S_o, _ = np.linalg.svd(block_o, full_matrices=False)
            _, S_w, _ = np.linalg.svd(block_w, full_matrices=False)

            extracted[i, j] = 1 if S_w[0] - S_o[0] > 0 else 0

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
# METHOD EXECUTOR
# =========================

def run_embedding_method(method):
    if method == "Block-SVD":
        return embed_watermark_svd_block, extract_watermark_svd_block
    elif method == "Improved DCT":
        return embed_watermark_dct_multi, extract_watermark_dct_multi
    elif method == "Improved DWT":
        return embed_watermark_dwt_ll, extract_watermark_dwt_ll
    elif method == "DCT-DWT":
        return embed_watermark_dct_dwt, extract_watermark_dct_dwt
    elif method == "DWT-DFT":
        return embed_watermark_dwt_dft, extract_watermark_dwt_dft
    elif method == "DCT-SVD":
        return embed_watermark_dct_svd, extract_watermark_dct_svd
    elif method == "DWT-SVD":
        return embed_watermark_dwt_svd, extract_watermark_dwt_svd
    else:
        return None, None


# =========================
# SIDEBAR SETTINGS
# =========================

st.sidebar.header(t["settings"])

domain = st.sidebar.selectbox(t["domain"], domain_options[lang])
attack_type = st.sidebar.selectbox(t["attack"], attack_options[lang])

method_options = [
    "Block-SVD",
    "Improved DCT",
    "Improved DWT",
    "DCT-DWT",
    "DWT-DFT",
    "DCT-SVD",
    "DWT-SVD"
]

selected_method = st.sidebar.selectbox(t["embedding_method"], method_options)

channel_options_display = [
    t["grayscale"],
    t["red_channel"],
    t["green_channel"],
    t["blue_channel"]
]

channel_mode_display = st.sidebar.selectbox(
    t["embedding_channel"],
    channel_options_display
)

channel_mapping = {
    t["grayscale"]: "Grayscale",
    t["red_channel"]: "Red",
    t["green_channel"]: "Green",
    t["blue_channel"]: "Blue"
}

channel_mode = channel_mapping[channel_mode_display]

recommended_alpha = predict_alpha_by_domain(domain)

st.sidebar.markdown("---")
st.sidebar.write(f"### {t['adaptive_alpha']}")

alpha_mode = st.sidebar.radio(t["alpha_selection"], [t["recommended"], t["manual"]])

if alpha_mode == t["recommended"]:
    predicted_alpha = recommended_alpha
else:
    predicted_alpha = st.sidebar.slider(t["alpha_value"], 5, 50, recommended_alpha, step=5)

st.sidebar.markdown("---")

attack_mode_display = st.sidebar.radio(
    t["attack_mode"],
    [t["single_attack"], t["combined_attack"]]
)

attack_mode = "Combined Attack" if attack_mode_display == t["combined_attack"] else "Single Attack"

combined_attack_options_display = [
    t["jpeg_noise"],
    t["jpeg_blur"],
    t["jpeg_noise_blur"],
    t["jpeg_noise_rotation"]
]

combined_attack_mapping = {
    t["jpeg_noise"]: "JPEG + Noise",
    t["jpeg_blur"]: "JPEG + Blur",
    t["jpeg_noise_blur"]: "JPEG + Noise + Blur",
    t["jpeg_noise_rotation"]: "JPEG + Noise + Rotation"
}

combined_attack_display = None
combined_attack = None

if attack_mode == "Combined Attack":
    combined_attack_display = st.sidebar.selectbox(
        t["combined_attack_type"],
        combined_attack_options_display
    )
    combined_attack = combined_attack_mapping[combined_attack_display]

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

st.sidebar.markdown("---")

max_images = st.sidebar.slider(t["max_images"], 1, 500, 50)

uploaded_files = st.file_uploader(
    t["upload_host"],
    type=["png", "jpg", "jpeg"],
    key="host_upload",
    accept_multiple_files=True
)

if uploaded_files:
    uploaded_files = uploaded_files[:max_images]

watermark_type = st.sidebar.radio(
    t["watermark_type"],
    [t["default_watermark"], t["upload_logo"], t["text_watermark"]]
)

watermark_file = None
watermark_text = ""

if watermark_type == t["upload_logo"]:
    watermark_file = st.file_uploader(t["upload_wm"], type=["png", "jpg", "jpeg"], key="watermark_upload")
elif watermark_type == t["text_watermark"]:
    watermark_text = st.sidebar.text_input(t["enter_watermark_text"], value="AV")

with st.expander(t["metric_exp"]):
    st.write(f"**PSNR:** {t['psnr_exp']}")
    st.write(f"**SSIM:** {t['ssim_exp']}")
    st.write(f"**BER:** {t['ber_exp']}")
    st.write(f"**Correlation:** {t['corr_exp']}")

# =========================
# MAIN PROCESS
# =========================

if uploaded_files:
    uploaded_file = uploaded_files[0]

    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_rgb = cv2.resize(img_rgb, (512, 512))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    host_image = get_host_channel(img_rgb, channel_mode)

    if watermark_type == t["upload_logo"] and watermark_file is not None:
        wm_img = Image.open(watermark_file).convert("L")
        wm_img = np.array(wm_img)
        wm_img = cv2.resize(wm_img, (32, 32))
        _, watermark_binary = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)
    elif watermark_type == t["text_watermark"]:
        watermark_binary = create_text_watermark(watermark_text)
    else:
        watermark_binary = create_default_watermark()

    embed_fn, extract_fn = run_embedding_method(selected_method)

    if embed_fn is None or extract_fn is None:
        st.error("Unknown embedding method selected.")
        st.stop()

    watermarked = embed_fn(host_image, watermark_binary, alpha=predicted_alpha)

    attacked = apply_selected_attack(
        watermarked,
        attack_mode,
        attack_type,
        attack_param,
        combined_attack
    )

    extracted = extract_fn(host_image, attacked, watermark_binary.shape)

    psnr_val = calculate_psnr(host_image, watermarked)
    ssim_val = calculate_ssim(host_image, watermarked)
    ber_val = calculate_ber(watermark_binary, extracted)
    corr_val = calculate_correlation(watermark_binary, extracted)

    score_val = calculate_score(psnr_val, ssim_val, ber_val, corr_val, domain)

    comparison_results = []
    extraction_gallery = []

    for method_name in method_options:
        try:
            embed_compare, extract_compare = run_embedding_method(method_name)

            if embed_compare is None or extract_compare is None:
                continue

            wm_compare = embed_compare(host_image, watermark_binary, alpha=predicted_alpha)

            attacked_compare = apply_selected_attack(
                wm_compare,
                attack_mode,
                attack_type,
                attack_param,
                combined_attack
            )

            extracted_compare = extract_compare(host_image, attacked_compare, watermark_binary.shape)

            psnr_compare = calculate_psnr(host_image, wm_compare)
            ssim_compare = calculate_ssim(host_image, wm_compare)
            ber_compare = calculate_ber(watermark_binary, extracted_compare)
            corr_compare = calculate_correlation(watermark_binary, extracted_compare)

            score_compare = calculate_score(psnr_compare, ssim_compare, ber_compare, corr_compare, domain)

            comparison_results.append({
                "Method": method_name,
                "PSNR": round(psnr_compare, 4),
                "SSIM": round(ssim_compare, 4),
                "BER": round(ber_compare, 4),
                "Correlation": round(corr_compare, 4),
                "Score": score_compare
            })

            extraction_gallery.append({
                "Method": method_name,
                "Extracted": extracted_compare
            })

        except Exception:
            comparison_results.append({
                "Method": method_name,
                "PSNR": np.nan,
                "SSIM": np.nan,
                "BER": np.nan,
                "Correlation": np.nan,
                "Score": np.nan
            })

    comparison_df = pd.DataFrame(comparison_results)
    valid_comparison_df = comparison_df.dropna(subset=["Score"])

    if not valid_comparison_df.empty:
        best_row = valid_comparison_df.loc[valid_comparison_df["Score"].idxmax()]
        recommended_method = best_row["Method"]
        recommended_score = best_row["Score"]
    else:
        recommended_method = selected_method
        recommended_score = score_val

    attack_display_name = combined_attack_display if attack_mode == "Combined Attack" else attack_type

    watermarked_display = reconstruct_display_image(img_rgb, watermarked, channel_mode)
    attacked_display = reconstruct_display_image(img_rgb, attacked, channel_mode)

    st.subheader(t["adaptive_decision"])
    st.write(f"{t['selected_domain']}: **{domain}**")
    st.write(f"{t['recommended_alpha']}: **{recommended_alpha}**")
    st.write(f"{t['selected_alpha']}: **{predicted_alpha}**")
    st.write(f"{t['method']}: **{selected_method}**")
    st.write(f"{t['embedding_channel']}: **{channel_mode_display}**")

    st.success(f"{t['recommended_method']}: {recommended_method} | Score = {recommended_score}")
    st.caption(f"{t['recommendation_reason']}: {t['highest_score_reason']}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("PSNR", f"{psnr_val:.2f} dB")
    col2.metric("SSIM", f"{ssim_val:.4f}")
    col3.metric("BER", f"{ber_val:.4f}")
    col4.metric("Correlation", f"{corr_val:.4f}")
    col5.metric("Score", f"{score_val:.4f}")

    st.subheader(t["images"])

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.image(img_rgb if channel_mode != "Grayscale" else img_gray, caption=t["original"], clamp=True)
    with c2:
        st.image(watermark_binary * 255, caption=t["input_wm"], clamp=True)
    with c3:
        st.image(watermarked_display, caption=f"{t['watermarked']} α={predicted_alpha}", clamp=True)
    with c4:
        st.image(attacked_display, caption=f"{t['after_attack']}: {attack_display_name}", clamp=True)
    with c5:
        st.image(extracted * 255, caption=t["extracted"], clamp=True)

    st.subheader(t["result_table"])

    result_df = pd.DataFrame({
        t["metric"]: [
            "Domain", "Method", "Channel", "Attack Mode", "Recommended Alpha",
            "Selected Alpha", "Attack", "PSNR", "SSIM", "BER", "Correlation", "Score"
        ],
        t["value"]: [
            domain, selected_method, channel_mode, attack_mode, recommended_alpha,
            predicted_alpha, attack_display_name, round(psnr_val, 4), round(ssim_val, 4),
            round(ber_val, 4), round(corr_val, 4), score_val
        ]
    })

    st.dataframe(result_df, use_container_width=True)

    if len(uploaded_files) > 1:
        batch_results = []

        watermarked_zip_buffer = io.BytesIO()
        attacked_zip_buffer = io.BytesIO()
        extracted_zip_buffer = io.BytesIO()

        progress_text = st.empty()
        progress_bar = st.progress(0)

        with zipfile.ZipFile(watermarked_zip_buffer, "w", zipfile.ZIP_DEFLATED) as watermarked_zip, \
             zipfile.ZipFile(attacked_zip_buffer, "w", zipfile.ZIP_DEFLATED) as attacked_zip, \
             zipfile.ZipFile(extracted_zip_buffer, "w", zipfile.ZIP_DEFLATED) as extracted_zip:

            total_files = len(uploaded_files)

            for idx, file in enumerate(uploaded_files):
                try:
                    progress_text.write(f"Processing image {idx + 1}/{total_files}: {file.name}")

                    image_b = Image.open(file).convert("RGB")
                    img_rgb_b = np.array(image_b)
                    img_rgb_b = cv2.resize(img_rgb_b, (512, 512))
                    img_gray_b = cv2.cvtColor(img_rgb_b, cv2.COLOR_RGB2GRAY)
                    host_image_b = get_host_channel(img_rgb_b, channel_mode)

                    watermarked_b = embed_fn(host_image_b, watermark_binary, alpha=predicted_alpha)

                    attacked_b = apply_selected_attack(
                        watermarked_b,
                        attack_mode,
                        attack_type,
                        attack_param,
                        combined_attack
                    )

                    extracted_b = extract_fn(host_image_b, attacked_b, watermark_binary.shape)

                    psnr_b = calculate_psnr(host_image_b, watermarked_b)
                    ssim_b = calculate_ssim(host_image_b, watermarked_b)
                    ber_b = calculate_ber(watermark_binary, extracted_b)
                    corr_b = calculate_correlation(watermark_binary, extracted_b)
                    score_b = calculate_score(psnr_b, ssim_b, ber_b, corr_b, domain)

                    clean_name = safe_filename(file.name)

                    watermarked_b_display = reconstruct_display_image(img_rgb_b, watermarked_b, channel_mode)
                    attacked_b_display = reconstruct_display_image(img_rgb_b, attacked_b, channel_mode)

                    add_image_to_zip(watermarked_zip, "watermarked_images", f"{clean_name}_watermarked", watermarked_b_display)
                    add_image_to_zip(attacked_zip, "attacked_images", f"{clean_name}_attacked", attacked_b_display)
                    add_image_to_zip(extracted_zip, "extracted_watermarks", f"{clean_name}_extracted_watermark", extracted_b * 255)

                    batch_results.append({
                        "Image": file.name,
                        "PSNR": round(psnr_b, 4),
                        "SSIM": round(ssim_b, 4),
                        "BER": round(ber_b, 4),
                        "Correlation": round(corr_b, 4),
                        "Score": score_b
                    })

                except Exception:
                    batch_results.append({
                        "Image": file.name,
                        "PSNR": np.nan,
                        "SSIM": np.nan,
                        "BER": np.nan,
                        "Correlation": np.nan,
                        "Score": np.nan
                    })

                progress_bar.progress((idx + 1) / total_files)

        progress_text.write(t["batch_completed"])

        watermarked_zip_buffer.seek(0)
        attacked_zip_buffer.seek(0)
        extracted_zip_buffer.seek(0)

        batch_df = pd.DataFrame(batch_results)

        st.markdown("---")
        st.subheader(t["batch_summary"])
        st.dataframe(batch_df, use_container_width=True)

        avg_df = batch_df[["PSNR", "SSIM", "BER", "Correlation", "Score"]].mean(numeric_only=True)

        avg_col1, avg_col2, avg_col3, avg_col4, avg_col5 = st.columns(5)
        avg_col1.metric("Avg PSNR", f"{avg_df['PSNR']:.2f} dB")
        avg_col2.metric("Avg SSIM", f"{avg_df['SSIM']:.4f}")
        avg_col3.metric("Avg BER", f"{avg_df['BER']:.4f}")
        avg_col4.metric("Avg Correlation", f"{avg_df['Correlation']:.4f}")
        avg_col5.metric("Avg Score", f"{avg_df['Score']:.4f}")

        st.subheader(t["batch_score_chart"])
        batch_chart_df = batch_df[["Image", "Score"]].set_index("Image")
        st.bar_chart(batch_chart_df)

        st.subheader(t["download_batch"])

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            st.download_button(
                label=t["download_watermarked_zip"],
                data=watermarked_zip_buffer.getvalue(),
                file_name="watermarked_images.zip",
                mime="application/zip"
            )

        with dl_col2:
            st.download_button(
                label=t["download_attacked_zip"],
                data=attacked_zip_buffer.getvalue(),
                file_name="attacked_images.zip",
                mime="application/zip"
            )

        with dl_col3:
            st.download_button(
                label=t["download_extracted_zip"],
                data=extracted_zip_buffer.getvalue(),
                file_name="extracted_watermarks.zip",
                mime="application/zip"
            )

    st.markdown("---")
    st.subheader(t["comparison_title"])
    st.write(t["comparison_desc"])
    st.dataframe(comparison_df, use_container_width=True)

    st.subheader(t["imperceptibility"])

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write(t["psnr_chart"])
        psnr_chart_df = comparison_df[["Method", "PSNR"]].set_index("Method")
        st.bar_chart(psnr_chart_df)

    with chart_col2:
        st.write(t["ssim_chart"])
        ssim_chart_df = comparison_df[["Method", "SSIM"]].set_index("Method")
        st.bar_chart(ssim_chart_df)

    st.subheader(t["robustness"])

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.write(t["ber_chart"])
        ber_chart_df = comparison_df[["Method", "BER"]].set_index("Method")
        st.bar_chart(ber_chart_df)

    with chart_col4:
        st.write(t["corr_chart"])
        corr_chart_df = comparison_df[["Method", "Correlation"]].set_index("Method")
        st.bar_chart(corr_chart_df)

    st.subheader(t["score_comparison"])

    score_chart_df = comparison_df[["Method", "Score"]].set_index("Method")
    st.bar_chart(score_chart_df)

    st.subheader(t["visual_gallery"])

    if extraction_gallery:
        gallery_cols = st.columns(len(extraction_gallery))

        for idx, item in enumerate(extraction_gallery):
            with gallery_cols[idx]:
                st.image(item["Extracted"] * 255, caption=item["Method"], clamp=True)

else:
    st.info(t["info"])
