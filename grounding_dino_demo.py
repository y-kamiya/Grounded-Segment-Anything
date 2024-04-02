from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
from pathlib import Path


CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
TEXT_PROMPT = "eye . eyebrow . mouth . nose . hair . ear . arm . neck . skirt . accessory . body"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

IMAGE_DIR = Path("data/images")

model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

def main(path: Path):
    image_source, image = load_image(path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device=DEVICE,
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(f"outputs/groundingdino/{path.name}", annotated_frame)


for path in IMAGE_DIR.iterdir():
    main(path)
