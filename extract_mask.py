import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import box_convert
import supervision as sv
import time

import sys
sys.path.append('/home/user/gradflow-gs/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# ----------- Config ------------
TEXT_PROMPT = ["a blue and white Gundam mecha model with large blue wings on the right side of the couch"]
IMG_PATH = "data/3dovs/sofa/images"  
OUTPUT_DIR = Path("Outputs/sofa")
SAM2_CHECKPOINT = "/home/user/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/user/checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------- Build Models ------------
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# ----------- Core Function ------------
def process_image(image_path: Path, text_prompt: str):
    # print(f"[INFO] Processing: {image_path.name}")
    image_source, image = load_image(str(image_path))
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    h, w, _ = image_source.shape
    base_name = image_path.stem + '.jpg'    # 统一保存为 xx 格式
    mask_save_path = os.path.join(OUTPUT_DIR, 'masks', text_prompt, base_name)
    anno_save_path = os.path.join(OUTPUT_DIR, 'annotated', text_prompt, base_name)

    if boxes.shape[0] == 0:
        print(f"[INFO] No detection in {base_name}")
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(mask_save_path), empty_mask)

        blank_anno = image_source.copy()
        cv2.putText(blank_anno, "No detection", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imwrite(str(anno_save_path), blank_anno)
        return

    # 保留最高置信度的框
    top_idx = torch.argmax(confidences)
    boxes = boxes[top_idx].unsqueeze(0)
    box_xyxy = box_convert(
        boxes=boxes * torch.tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy"
    ).numpy()

    masks, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_xyxy,
        multimask_output=False
    )

    if masks.ndim == 4:
        masks = masks.squeeze(0).squeeze(0)
    else:
        masks = masks.squeeze(0)

    binary_mask = (masks * 255).astype(np.uint8)
    cv2.imwrite(str(mask_save_path), binary_mask)
    # print(f"[INFO] Saved mask: {mask_save_path.name}")

    # === 可视化并保存 annotated_frame ===
    detections = sv.Detections(
        xyxy=box_xyxy,
        mask=masks.astype(bool)[None, :, :],
        class_id=np.array([0])  # 单类别编号
    )

    labels = [f"{labels[top_idx]} {confidences[top_idx]:.2f}"]

    frame = image_source.copy()
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = mask_annotator.annotate(scene=frame, detections=detections)
    cv2.imwrite(str(anno_save_path), frame)
    # print(f"[INFO] Saved annotation: {anno_save_path.name}")

# ----------- Run Batch ------------
def batch_process_all_images(folder: str, text_prompt: str):
    image_paths = sorted(Path(folder).glob("*.jpg"))  # 可改为 "*.png" 或其他格式
    for img_path in image_paths:
        process_image(img_path, text_prompt)

# ----------- Entry Point ------------
if __name__ == "__main__":
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()

    start_time = time.time()
    for text_prompt in TEXT_PROMPT:
        print(f"\nProcessing prompt: '{text_prompt}' ...\n")
        annotated_floder = os.path.join(OUTPUT_DIR, 'annotated', text_prompt)
        masks_floder = os.path.join(OUTPUT_DIR, 'masks', text_prompt)
        os.makedirs(masks_floder, exist_ok=True)
        os.makedirs(annotated_floder, exist_ok=True)

        batch_process_all_images(IMG_PATH, text_prompt)

    total_time = time.time() - start_time
    print(f'\nAll completed in {total_time:.2f} seconds.\n')
