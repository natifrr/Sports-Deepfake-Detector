from transformers import pipeline
from PIL import Image, ExifTags
import numpy as np
import cv2

classifier = pipeline(
    "image-classification",
    model="prithivMLmods/deepfake-detector-model-v1",
    device=-1
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def compute_blur_score(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    return faces

def extract_metadata(image):
    metadata = {}
    exif = image.getexif()

    if not exif:
        return metadata

    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        metadata[tag_name] = value

    return metadata


def analyze_metadata(image):
    metadata = extract_metadata(image)
    warnings = []
    suspicion_score = 0

    if not metadata:
        warnings.append("No EXIF metadata found")
        suspicion_score += 1
        return {
            "metadata_found": False,
            "warnings": warnings,
            "suspicion_score": suspicion_score,
            "summary": "No EXIF metadata was found. This is common for screenshots, social uploads, and AI-generated exports."
        }

    software_value = str(metadata.get("Software", "")).lower()
    make_value = str(metadata.get("Make", "")).strip()
    model_value = str(metadata.get("Model", "")).strip()
    datetime_original = str(metadata.get("DateTimeOriginal", "")).strip()
    datetime_general = str(metadata.get("DateTime", "")).strip()

    suspicious_software_keywords = [
        "photoshop",
        "gimp",
        "canva",
        "snapseed",
        "facetune",
        "lightroom",
        "midjourney",
        "stable diffusion",
        "dall",
        "comfyui"
    ]

    if software_value:
        for keyword in suspicious_software_keywords:
            if keyword in software_value:
                warnings.append(f"Edited/exported with software: {metadata.get('Software')}")
                suspicion_score += 2
                break

    if not make_value and not model_value:
        warnings.append("No camera make/model found")
        suspicion_score += 1

    if datetime_original and datetime_general and datetime_original != datetime_general:
        warnings.append("DateTimeOriginal and DateTime differ")
        suspicion_score += 1

    if not datetime_original and not datetime_general:
        warnings.append("No image timestamp found")
        suspicion_score += 1

    summary_parts = []
    summary_parts.append("EXIF metadata found")

    if make_value or model_value:
        summary_parts.append(f"Camera: {make_value} {model_value}".strip())

    if datetime_original:
        summary_parts.append(f"Captured: {datetime_original}")
    elif datetime_general:
        summary_parts.append(f"Timestamp: {datetime_general}")

    if software_value:
        summary_parts.append(f"Software: {metadata.get('Software')}")

    return {
        "metadata_found": True,
        "warnings": warnings,
        "suspicion_score": suspicion_score,
        "summary": " | ".join(summary_parts)
    }

def detect_deepfake(image):
    if image is None:
        return "No image uploaded."

    original_image = image
    metadata_info = analyze_metadata(original_image)

    image = image.convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    blur_score = compute_blur_score(image_np)
    faces = detect_faces(image_np)
    face_count = len(faces)

    quality_warnings = []
    
    if metadata_info["suspicion_score"] >= 2:
        quality_warnings.append("suspicious metadata")

    if width < 512 or height < 512:
        quality_warnings.append("low resolution")
    if blur_score < 100:
        quality_warnings.append("blurry image")
    if face_count == 0:
        quality_warnings.append("no clear face detected")

    results = classifier(image)
    top_result = results[0]
    raw_label = top_result["label"]
    raw_score = top_result["score"] * 100

    label_lower = raw_label.lower()

    if (
        face_count == 0
        or blur_score < 100
        or width < 512
        or height < 512
        or raw_score < 75
    ):
        verdict = "Inconclusive"
        explanation = (
            "The detector does not have enough confidence to make a reliable call "
            "on this image."
        )
    else:
        if (
            ("fake" in label_lower or "deepfake" in label_lower)
            and metadata_info["suspicion_score"] >= 2
        ):
            verdict = "Likely Synthetic"
            explanation = (
                "The model prediction and metadata signals both suggest possible "
                "manipulation or synthetic origin."
            )
        elif "fake" in label_lower or "deepfake" in label_lower:
            verdict = "Possibly Synthetic"
            explanation = (
                "The visual model flagged the image, but metadata evidence is limited."
            )
        else:
            verdict = "Likely Authentic"
            explanation = (
                "The model found patterns more consistent with an authentic image."
            )

    warning_text = ", ".join(quality_warnings) if quality_warnings else "none"
    metadata_warning_text = (
        ", ".join(metadata_info["warnings"])
        if metadata_info["warnings"]
        else "none"
    )

    return f"""
Prediction: {verdict}
Model Label: {raw_label}
Model Confidence: {raw_score:.2f}%

Image Size: {width} x {height}
Blur Score: {blur_score:.2f}
Faces Detected: {face_count}
Warnings: {warning_text}

Metadata Found: {metadata_info['metadata_found']}
Metadata Score: {metadata_info['suspicion_score']}
Metadata Warnings: {metadata_warning_text}
Metadata Summary: {metadata_info['summary']}

Explanation: {explanation}
"""