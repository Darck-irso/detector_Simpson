from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tensorflow.keras.models import load_model
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "modelo_final_simpsons.keras"
MODEL_H5_PATH = MODELS_DIR / "modelo_final_simpsons.h5"
YOLO_PATH = MODELS_DIR / "yolo_simpsons_best.pt"
CLASSES_PATH = MODELS_DIR / "class_indices_simpsons.json"
CONFIG_PATH = MODELS_DIR / "pipeline_config.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


def require_artifacts() -> None:
    if not MODEL_PATH.is_file() and not MODEL_H5_PATH.is_file():
        raise FileNotFoundError(
            "Falta models/modelo_final_simpsons.keras "
            "(o models/modelo_final_simpsons.h5)."
        )

    missing = [
        path.name
        for path in (YOLO_PATH, CLASSES_PATH, CONFIG_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos en la carpeta models: " + ", ".join(missing)
        )


require_artifacts()

with CLASSES_PATH.open("r", encoding="utf-8") as file:
    CLASS_INDICES = json.load(file)

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    PIPELINE_CONFIG = json.load(file)

IDX_TO_CLASS = {
    int(index): class_name
    for class_name, index in CLASS_INDICES.items()
}
CLASS_TO_IDX = {
    class_name: int(index)
    for class_name, index in CLASS_INDICES.items()
}

CLASSIFIER_PATH = MODEL_PATH if MODEL_PATH.is_file() else MODEL_H5_PATH
CLASSIFIER = load_model(CLASSIFIER_PATH, compile=False)
DETECTOR = YOLO(str(YOLO_PATH))
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

if int(CLASSIFIER.output_shape[-1]) != len(IDX_TO_CLASS):
    raise ValueError(
        "El número de salidas del clasificador no coincide con el JSON de clases."
    )


def config_float(name: str, default: float) -> float:
    return float(PIPELINE_CONFIG.get(name, default))


def config_int(name: str, default: int) -> int:
    return int(PIPELINE_CONFIG.get(name, default))


FRAME_INTERVAL = config_int("frame_interval", 3)
PADDING_FACTOR = config_float("padding_factor", 0.05)
MIN_AREA_RATIO = config_float("min_area_ratio", 0.001)
MAX_AREA_RATIO = config_float("max_area_ratio", 0.90)
MIN_BOX_WIDTH = config_int("min_box_width", 30)
MIN_BOX_HEIGHT = config_int("min_box_height", 30)
TOP2_MARGIN = config_float("top2_margin", 0.15)

IOU_TRACK_MIN = config_float("iou_track_minimo", 0.15)
MAX_TRACK_GAP = FRAME_INTERVAL * 4
MIN_TRACK_APPEARANCES = config_int("min_apariciones_track", 5)
SCENE_CHANGE_THRESHOLD = 45.0

TEMPORAL_CONFIDENCE = config_float("umbral_consenso_temporal", 0.52)
TEMPORAL_MARGIN = config_float("margen_consenso_temporal", 0.20)
MIN_INITIAL_APPEARANCES = config_int("min_apariciones_iniciales", 4)
LABEL_CHANGE_CONFIRMATIONS = config_int("cambios_etiqueta_requeridos", 5)
PROBABILITIES_EMA_ALPHA = config_float("ema_probabilidades_alpha", 0.35)
BBOX_EMA_ALPHA = config_float("ema_bbox_alpha", 0.45)
MIN_PRESENCE_EVIDENCE = config_int("min_detecciones_presencia", 8)

MODERATE_CONFIDENCE = config_float("confianza_observacion_consenso", 0.25)
MODERATE_MARGIN = config_float("margen_observacion_consenso", 0.05)
MIN_MODERATE_OBSERVATIONS = config_int("min_observaciones_consenso_debil", 20)
WEAK_CONSENSUS_CONFIDENCE = config_float("umbral_consenso_debil", 0.40)
WEAK_CONSENSUS_MARGIN = config_float("margen_consenso_debil", 0.12)
MAX_ANNOTATION_AGE = FRAME_INTERVAL * 4


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def predict_character(crop_bgr: np.ndarray, confidence_threshold: float) -> dict:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    input_height = int(CLASSIFIER.input_shape[1])
    input_width = int(CLASSIFIER.input_shape[2])

    model_input = cv2.resize(crop_rgb, (input_width, input_height))
    model_input = model_input.astype("float32") / 255.0
    probabilities = CLASSIFIER.predict(
        np.expand_dims(model_input, axis=0),
        verbose=0,
    )[0]

    order = np.argsort(probabilities)[::-1]
    top1_index = int(order[0])
    top2_index = int(order[1])
    top1_confidence = float(probabilities[top1_index])
    top2_confidence = float(probabilities[top2_index])
    margin = top1_confidence - top2_confidence

    return {
        "character": IDX_TO_CLASS[top1_index],
        "confidence": top1_confidence,
        "margin": margin,
        "accepted": (
            top1_confidence >= float(confidence_threshold)
            and margin >= TOP2_MARGIN
        ),
        "crop_rgb": crop_rgb,
        "probabilities": probabilities.astype("float32"),
    }


def detect_candidates(
    frame: np.ndarray,
    yolo_confidence: float,
    classifier_confidence: float,
) -> dict:
    height, width = frame.shape[:2]
    frame_area = max(height * width, 1)
    candidates = []
    rejected_area = 0

    detections = DETECTOR(
        frame,
        conf=float(yolo_confidence),
        iou=0.45,
        verbose=False,
    )
    boxes = detections[0].boxes
    total_yolo = 0 if boxes is None else len(boxes)

    if boxes is None:
        return {
            "candidates": candidates,
            "total_yolo": total_yolo,
            "rejected_area": rejected_area,
        }

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
            rejected_area += 1
            continue

        area_ratio = box_width * box_height / frame_area
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            rejected_area += 1
            continue

        padding_x = int(box_width * PADDING_FACTOR)
        padding_y = int(box_height * PADDING_FACTOR)
        roi_x1 = max(0, x1 - padding_x)
        roi_y1 = max(0, y1 - padding_y)
        roi_x2 = min(width, x2 + padding_x)
        roi_y2 = min(height, y2 + padding_y)

        roi_width = roi_x2 - roi_x1
        roi_height = roi_y2 - roi_y1
        if roi_width <= 0 or roi_height <= 0:
            rejected_area += 1
            continue
        if roi_width * roi_height / frame_area > MAX_AREA_RATIO:
            rejected_area += 1
            continue

        crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if crop.size == 0:
            rejected_area += 1
            continue

        prediction = predict_character(crop, classifier_confidence)
        candidates.append({
            **prediction,
            "yolo_confidence": float(box.conf[0]),
            "bbox": [roi_x1, roi_y1, roi_x2, roi_y2],
        })

    return {
        "candidates": candidates,
        "total_yolo": int(total_yolo),
        "rejected_area": int(rejected_area),
    }


def calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return intersection / (area_a + area_b - intersection)


def scene_change(frame: np.ndarray, reference: np.ndarray | None) -> tuple:
    thumbnail = cv2.resize(frame, (64, 36))
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    if reference is None:
        return gray, 0.0
    return gray, float(cv2.absdiff(gray, reference).mean())


def create_track(track_id: int, bbox: list[int], probabilities: np.ndarray, frame_id: int) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return {
        "track_id": track_id,
        "bbox_ema": np.asarray(bbox, dtype=np.float32),
        "probabilities_ema": np.zeros_like(probabilities),
        "last_frame": frame_id,
        "active": True,
        "appearances": 0,
        "strong_observations": 0,
        "consensus_observations": 0,
        "stable_label": None,
        "stable_confidence": 0.0,
        "stable_margin": 0.0,
        "evidence_mode": None,
        "candidate_label": None,
        "candidate_confirmations": 0,
    }


def track_has_strong_evidence(track: dict) -> bool:
    return (
        track["strong_observations"] >= MIN_INITIAL_APPEARANCES
        and track["stable_confidence"] >= TEMPORAL_CONFIDENCE
        and track["stable_margin"] >= TEMPORAL_MARGIN
    )


def track_has_temporal_evidence(track: dict) -> bool:
    return (
        track["consensus_observations"] >= MIN_MODERATE_OBSERVATIONS
        and track["stable_confidence"] >= WEAK_CONSENSUS_CONFIDENCE
        and track["stable_margin"] >= WEAK_CONSENSUS_MARGIN
    )


def update_track(track: dict, candidate: dict, frame_id: int) -> None:
    bbox = np.asarray(candidate["bbox"], dtype=np.float32)
    probabilities = np.asarray(candidate["probabilities"], dtype=np.float32)

    if track["appearances"] <= 1:
        track["bbox_ema"] = bbox.copy()
    else:
        track["bbox_ema"] = (
            BBOX_EMA_ALPHA * bbox
            + (1.0 - BBOX_EMA_ALPHA) * track["bbox_ema"]
        )

    order = np.argsort(probabilities)[::-1]
    instant_confidence = float(probabilities[order[0]])
    instant_margin = instant_confidence - float(probabilities[order[1]])
    consensus_observation = (
        candidate["accepted"]
        or (
            instant_confidence >= MODERATE_CONFIDENCE
            and instant_margin >= MODERATE_MARGIN
        )
    )

    if consensus_observation:
        if track["consensus_observations"] == 0:
            track["probabilities_ema"] = probabilities.copy()
        else:
            track["probabilities_ema"] = (
                PROBABILITIES_EMA_ALPHA * probabilities
                + (1.0 - PROBABILITIES_EMA_ALPHA)
                * track["probabilities_ema"]
            )
        track["consensus_observations"] += 1

    if candidate["accepted"]:
        track["strong_observations"] += 1

    track["last_frame"] = frame_id
    consensus_probabilities = (
        track["probabilities_ema"]
        if track["consensus_observations"] > 0
        else probabilities
    )
    order = np.argsort(consensus_probabilities)[::-1]
    label = IDX_TO_CLASS[int(order[0])]
    confidence = float(consensus_probabilities[order[0]])
    margin = confidence - float(consensus_probabilities[order[1]])

    strong_candidate = (
        track["strong_observations"] >= MIN_INITIAL_APPEARANCES
        and confidence >= TEMPORAL_CONFIDENCE
        and margin >= TEMPORAL_MARGIN
    )
    temporal_candidate = (
        track["consensus_observations"] >= MIN_MODERATE_OBSERVATIONS
        and confidence >= WEAK_CONSENSUS_CONFIDENCE
        and margin >= WEAK_CONSENSUS_MARGIN
    )
    if not (strong_candidate or temporal_candidate):
        return

    evidence_mode = "strong" if strong_candidate else "temporal"
    if track["stable_label"] is None:
        track["stable_label"] = label
        track["stable_confidence"] = confidence
        track["stable_margin"] = margin
        track["evidence_mode"] = evidence_mode
        return

    if label == track["stable_label"]:
        track["stable_confidence"] = confidence
        track["stable_margin"] = margin
        if strong_candidate:
            track["evidence_mode"] = "strong"
        track["candidate_label"] = None
        track["candidate_confirmations"] = 0
        return

    if track["candidate_label"] == label:
        track["candidate_confirmations"] += 1
    else:
        track["candidate_label"] = label
        track["candidate_confirmations"] = 1

    if (
        confidence >= track["stable_confidence"] + 0.08
        and track["candidate_confirmations"] >= LABEL_CHANGE_CONFIRMATIONS
    ):
        track["stable_label"] = label
        track["stable_confidence"] = confidence
        track["stable_margin"] = margin
        track["evidence_mode"] = evidence_mode
        track["candidate_label"] = None
        track["candidate_confirmations"] = 0


def assign_tracks(
    candidates: list[dict],
    tracks: dict[int, dict],
    frame_id: int,
    next_track_id: int,
) -> int:
    for track in tracks.values():
        if frame_id - track["last_frame"] > MAX_TRACK_GAP:
            track["active"] = False

    assigned_tracks = set()
    for candidate in candidates:
        best_track_id = None
        best_iou = 0.0
        for track_id, track in tracks.items():
            if (
                track_id in assigned_tracks
                or not track["active"]
                or frame_id - track["last_frame"] > MAX_TRACK_GAP
            ):
                continue
            iou = calculate_iou(candidate["bbox"], track["bbox_ema"])
            if iou >= IOU_TRACK_MIN and iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = next_track_id
            next_track_id += 1
            tracks[best_track_id] = create_track(
                best_track_id,
                candidate["bbox"],
                candidate["probabilities"],
                frame_id,
            )

        track = tracks[best_track_id]
        track["appearances"] += 1
        track["active"] = True
        update_track(track, candidate, frame_id)

        candidate["track_id"] = best_track_id
        candidate["stable_label"] = track["stable_label"]
        candidate["stable_confidence"] = track["stable_confidence"]
        candidate["accepted_consensus"] = (
            track["stable_label"] is not None
            and (track_has_strong_evidence(track) or track_has_temporal_evidence(track))
        )
        assigned_tracks.add(best_track_id)

    return next_track_id


def active_annotations(tracks: dict[int, dict], frame_id: int) -> list[dict]:
    annotations = []
    for track in tracks.values():
        if (
            not track["active"]
            or track["stable_label"] is None
            or frame_id - track["last_frame"] > MAX_ANNOTATION_AGE
            or not (track_has_strong_evidence(track) or track_has_temporal_evidence(track))
        ):
            continue

        annotations.append({
            "character": track["stable_label"],
            "confidence": track["stable_confidence"],
            "bbox": np.rint(track["bbox_ema"]).astype(int).tolist(),
        })
    return annotations


def draw_annotation(frame: np.ndarray, annotation: dict) -> None:
    x1, y1, x2, y2 = [int(value) for value in annotation["bbox"]]
    label = (
        f"{display_name(annotation['character'])} "
        f"({annotation['confidence']:.0%})"
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 90), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (40, 220, 90),
        2,
    )


def best_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf"]
        if bold
        else ["DejaVuSans.ttf", "arial.ttf"]
    )
    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_character_summary(items: dict[str, dict], output_path: Path) -> str | None:
    if not items:
        return None

    ordered_items = sorted(items.items(), key=lambda item: item[0])
    columns = min(3, len(ordered_items))
    rows = math.ceil(len(ordered_items) / columns)
    card_width = 330
    card_height = 315
    header_height = 92
    canvas = Image.new(
        "RGB",
        (columns * card_width + 40, rows * card_height + header_height + 25),
        "#111827",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 22),
        "Personajes reconocidos",
        fill="#F9FAFB",
        font=best_font(28, bold=True),
    )
    draw.text(
        (24, 58),
        f"{len(ordered_items)} personaje(s) único(s)",
        fill="#9CA3AF",
        font=best_font(17),
    )

    for index, (character, data) in enumerate(ordered_items):
        row, column = divmod(index, columns)
        left = 20 + column * card_width
        top = header_height + row * card_height
        right = left + card_width - 16
        bottom = top + card_height - 16
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=18,
            fill="#1F2937",
            outline="#374151",
            width=2,
        )

        crop = Image.fromarray(data["crop_rgb"]).convert("RGB")
        crop = ImageOps.contain(crop, (282, 205), Image.Resampling.LANCZOS)
        crop_background = Image.new("RGB", (282, 205), "#0B1220")
        crop_background.paste(
            crop,
            ((282 - crop.width) // 2, (205 - crop.height) // 2),
        )
        canvas.paste(crop_background, (left + 16, top + 16))

        draw.text(
            (left + 16, top + 230),
            display_name(character),
            fill="#F9FAFB",
            font=best_font(19, bold=True),
        )
        draw.text(
            (left + 16, top + 260),
            (
                f"Confianza {data['confidence']:.0%}  ·  "
                f"{data['observations']} observaciones"
            ),
            fill="#A7F3D0",
            font=best_font(14),
        )

    canvas.save(output_path, quality=92)
    return str(output_path)


def build_text_summary(
    file_type: str,
    characters: dict[str, dict],
    total_yolo: int,
    rejected_area: int,
    weak_predictions: int,
    frames: int | None = None,
) -> str:
    lines = [f"Resultado del análisis de {file_type}", ""]
    if frames is not None:
        lines.append(f"Frames procesados: {frames}")
    lines.extend([
        f"Detecciones iniciales de YOLO: {total_yolo}",
        f"Descartadas por tamaño o área: {rejected_area}",
        f"Predicciones individuales débiles: {weak_predictions}",
        f"Personajes únicos confirmados: {len(characters)}",
        "",
    ])

    if not characters:
        lines.append("No se encontraron personajes con evidencia suficiente.")
    else:
        lines.append("Personajes reconocidos:")
        for character, data in sorted(characters.items()):
            lines.append(
                f"- {display_name(character)}: "
                f"{data['confidence']:.1%} de confianza, "
                f"{data['observations']} observaciones"
            )
    return "\n".join(lines)


def process_image(
    image_path: str,
    yolo_confidence: float,
    classifier_confidence: float,
) -> tuple[str, str | None, str]:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError("No fue posible leer la imagen seleccionada.")

    analysis = detect_candidates(image, yolo_confidence, classifier_confidence)
    annotated = image.copy()
    characters = {}
    weak_predictions = 0

    for candidate in analysis["candidates"]:
        if not candidate["accepted"]:
            weak_predictions += 1
            continue

        character = candidate["character"]
        draw_annotation(
            annotated,
            {
                "character": character,
                "confidence": candidate["confidence"],
                "bbox": candidate["bbox"],
            },
        )
        previous = characters.get(character)
        if previous is None or candidate["confidence"] > previous["confidence"]:
            characters[character] = {
                "crop_rgb": candidate["crop_rgb"],
                "confidence": candidate["confidence"],
                "observations": 1,
            }
        elif previous is not None:
            previous["observations"] += 1

    output_dir = Path(tempfile.mkdtemp(prefix="simpsons_image_"))
    annotated_path = output_dir / "imagen_anotada.jpg"
    summary_path = output_dir / "personajes_reconocidos.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    summary_image = create_character_summary(characters, summary_path)
    summary_text = build_text_summary(
        "imagen",
        characters,
        analysis["total_yolo"],
        analysis["rejected_area"],
        weak_predictions,
    )
    return str(annotated_path), summary_image, summary_text


def run_ffmpeg(arguments: list[str], error_message: str) -> None:
    result = subprocess.run(
        [FFMPEG_BIN, "-y", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(error_message + "\n\n" + result.stderr[-1200:])


def process_video(
    video_path: str,
    yolo_confidence: float,
    classifier_confidence: float,
    progress: gr.Progress,
) -> tuple[str, str | None, str]:
    output_dir = Path(tempfile.mkdtemp(prefix="simpsons_video_"))
    converted_input = output_dir / "entrada_h264.mp4"
    temporary_video = output_dir / "video_anotado_temporal.mp4"
    final_video = output_dir / "video_anotado_h264.mp4"
    summary_path = output_dir / "personajes_reconocidos.jpg"

    progress(0.02, desc="Preparando video")
    run_ffmpeg(
        [
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(converted_input),
        ],
        "No fue posible convertir el video de entrada.",
    )

    capture = cv2.VideoCapture(str(converted_input))
    if not capture.isOpened():
        raise RuntimeError("No fue posible abrir el video convertido.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("El video tiene dimensiones inválidas.")

    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("No fue posible crear el video anotado temporal.")

    tracks = {}
    next_track_id = 0
    reference_scene = None
    current_annotations = []
    best_crops = {}
    repeated_counts = {}
    frame_id = 0
    total_yolo = 0
    rejected_area = 0
    weak_predictions = 0
    progress_interval = max(total_frames // 100, 1)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        annotated = frame.copy()
        if frame_id % FRAME_INTERVAL == 0:
            reference_scene, change_value = scene_change(frame, reference_scene)
            if change_value >= SCENE_CHANGE_THRESHOLD:
                for track in tracks.values():
                    track["active"] = False
                current_annotations = []

            analysis = detect_candidates(
                frame,
                yolo_confidence,
                classifier_confidence,
            )
            next_track_id = assign_tracks(
                analysis["candidates"],
                tracks,
                frame_id,
                next_track_id,
            )
            current_annotations = active_annotations(tracks, frame_id)
            total_yolo += analysis["total_yolo"]
            rejected_area += analysis["rejected_area"]
            weak_predictions += sum(
                not candidate["accepted"]
                for candidate in analysis["candidates"]
            )

            for candidate in analysis["candidates"]:
                if not candidate.get("accepted_consensus"):
                    continue
                character = candidate["stable_label"]
                character_index = CLASS_TO_IDX[character]
                character_score = float(
                    candidate["probabilities"][character_index]
                )
                repeated_counts[character] = repeated_counts.get(character, 0) + 1
                previous = best_crops.get(character)
                if previous is None or character_score > previous["crop_score"]:
                    best_crops[character] = {
                        "crop_rgb": candidate["crop_rgb"],
                        "crop_score": character_score,
                    }

        for annotation in current_annotations:
            draw_annotation(annotated, annotation)
        writer.write(annotated)
        frame_id += 1

        if total_frames > 0 and (
            frame_id % progress_interval == 0 or frame_id >= total_frames
        ):
            ratio = min(frame_id / total_frames, 1.0)
            progress(
                0.08 + ratio * 0.82,
                desc=f"Analizando video: {int(ratio * 100)}%",
            )

    capture.release()
    writer.release()

    confirmed_characters = {}
    for track in tracks.values():
        if track["stable_label"] is None:
            continue
        strong = track_has_strong_evidence(track)
        temporal = track_has_temporal_evidence(track)
        if not (strong or temporal):
            continue

        evidence = (
            track["strong_observations"]
            if strong
            else track["consensus_observations"]
        )
        if evidence < MIN_PRESENCE_EVIDENCE:
            continue

        character = track["stable_label"]
        previous = confirmed_characters.get(character)
        if previous is None or evidence > previous["observations"]:
            crop_data = best_crops.get(character)
            if crop_data is None:
                continue
            confirmed_characters[character] = {
                "crop_rgb": crop_data["crop_rgb"],
                "confidence": float(track["stable_confidence"]),
                "observations": int(evidence),
                "repeated_detections": int(repeated_counts.get(character, 0)),
            }

    progress(0.92, desc="Codificando video final con audio")
    run_ffmpeg(
        [
            "-i", str(temporary_video),
            "-i", str(converted_input),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(final_video),
        ],
        "No fue posible generar el video anotado final.",
    )

    summary_image = create_character_summary(confirmed_characters, summary_path)
    summary_text = build_text_summary(
        "video",
        confirmed_characters,
        total_yolo,
        rejected_area,
        weak_predictions,
        frames=frame_id,
    )
    progress(1.0, desc="Procesamiento completado")
    return str(final_video), summary_image, summary_text


def process_file(
    file_path: str | None,
    yolo_confidence: float,
    classifier_confidence: float,
    progress=gr.Progress(),
):
    if not file_path:
        return (
            gr.Image(value=None, visible=False),
            gr.Video(value=None, visible=False),
            None,
            "Debes seleccionar una imagen o un video.",
        )

    extension = Path(file_path).suffix.lower()
    try:
        if extension in IMAGE_EXTENSIONS:
            progress(0.15, desc="Analizando imagen")
            annotated, summary_image, summary_text = process_image(
                file_path,
                yolo_confidence,
                classifier_confidence,
            )
            progress(1.0, desc="Imagen procesada")
            return (
                gr.Image(value=annotated, visible=True),
                gr.Video(value=None, visible=False),
                summary_image,
                summary_text,
            )

        if extension in VIDEO_EXTENSIONS:
            video, summary_image, summary_text = process_video(
                file_path,
                yolo_confidence,
                classifier_confidence,
                progress,
            )
            return (
                gr.Image(value=None, visible=False),
                gr.Video(value=video, visible=True),
                summary_image,
                summary_text,
            )

        return (
            gr.Image(value=None, visible=False),
            gr.Video(value=None, visible=False),
            None,
            "Formato no compatible. Selecciona una imagen o un video.",
        )
    except Exception as error:
        return (
            gr.Image(value=None, visible=False),
            gr.Video(value=None, visible=False),
            None,
            f"No fue posible procesar el archivo:\n{error}",
        )


CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto;}
#hero {text-align: center; margin-bottom: 1rem;}
#process-button {min-height: 46px;}
"""

with gr.Blocks(css=CSS, title="Reconocimiento de personajes de Los Simpsons") as demo:
    gr.Markdown(
        """
        # Reconocimiento de personajes de Los Simpsons

        Carga una imagen o un video. YOLO localiza los personajes y el
        clasificador identifica cada región. En video, el consenso temporal
        estabiliza las predicciones entre frames.
        """,
        elem_id="hero",
    )

    file_input = gr.File(
        label="Imagen o video",
        file_types=["image", "video"],
        file_count="single",
        type="filepath",
    )

    with gr.Row():
        yolo_confidence_input = gr.Slider(
            minimum=0.10,
            maximum=0.90,
            value=config_float("confianza_yolo", 0.30),
            step=0.05,
            label="Confianza mínima YOLO",
        )
        classifier_confidence_input = gr.Slider(
            minimum=0.10,
            maximum=0.95,
            value=config_float("confianza_clasificador", 0.50),
            step=0.05,
            label="Confianza mínima del clasificador",
        )

    process_button = gr.Button(
        "Analizar archivo",
        variant="primary",
        elem_id="process-button",
    )

    annotated_image_output = gr.Image(
        label="Imagen anotada",
        visible=False,
        interactive=False,
    )
    annotated_video_output = gr.Video(
        label="Video anotado",
        visible=False,
        interactive=False,
        format="mp4",
    )

    with gr.Row():
        summary_image_output = gr.Image(
            label="Personajes únicos reconocidos",
            interactive=False,
        )
        summary_text_output = gr.Textbox(
            label="Resumen del análisis",
            lines=14,
            interactive=False,
        )

    process_button.click(
        fn=process_file,
        inputs=[
            file_input,
            yolo_confidence_input,
            classifier_confidence_input,
        ],
        outputs=[
            annotated_image_output,
            annotated_video_output,
            summary_image_output,
            summary_text_output,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        show_error=True,
    )
