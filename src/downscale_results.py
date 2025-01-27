import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from signal import SIGINT

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    batch_id = cfg.batch_id
    remove_images = cfg.downscale.remove_images
    remove_raws = cfg.downscale.remove_raws
    input_raw_dir = Path(cfg.paths.local_upload) / batch_id / "raw"
    input_image_dir = Path(cfg.paths.local_upload) / batch_id / "colorcorrected"
    output_image_dir = Path(cfg.paths.local_upload) / batch_id / "downscaled"
    output_image_dir.mkdir(parents=True, exist_ok=True)
    downscale_factor = cfg.downscale.factor

    for image_path in input_image_dir.glob("*.jpg"):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        new_height = int(height * downscale_factor)
        new_width = int(width * downscale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        output_image_path = output_image_dir / image_path.name
        cv2.imwrite(str(output_image_path), resized_image)
        log.info(f"Saved: {output_image_path}")

    # Remove image from source directory
    if remove_images:
        for image_path in input_image_dir.glob("*.jpg"):
            image_path.unlink()
            log.info(f"Removed JPG: {image_path}")

    # Remove image from source directory
    if remove_raws:
        for image_path in input_raw_dir.glob("*.RAW"):
            image_path.unlink()
            log.info(f"Removed RAW: {image_path}")

if __name__ == "__main__":
    main()
