import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import signal
from signal import SIGINT

log = logging.getLogger(__name__)

# Signal handler setup for graceful termination
def setup_signal_handler():
    def signal_handler(signal_received, frame):
        log.info("Interrupt signal received. Cleaning up...")
        raise KeyboardInterrupt

    signal.signal(SIGINT, signal_handler)


class FileManager:
    """Handles file-related operations like copying and filtering."""

    @staticmethod
    def copy_file(src: str, dest: str):
        if not os.path.exists(dest) or os.path.getsize(src) != os.path.getsize(dest):
            shutil.copy2(src, dest)
            log.info(f"Copied {src} to {dest}")
        else:
            log.info(f"Skipped {src}, already present at destination with matching size")

    @staticmethod
    def copy_files_in_parallel(src_dir: Path, dest_dir: Path, files: list, max_workers=8):
        if not dest_dir.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Copying {len(files)} files to {dest_dir}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(FileManager.copy_file, str(file), str(dest_dir / file.name)) for file in files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error copying file: {e}")

    @staticmethod
    def get_sampled_files(raw_dir: Path, sample_size: int, strategy: str = "random") -> list:
        jpg_files = sorted(list(raw_dir.glob("*.jpg")))
        if sample_size and len(jpg_files) > sample_size:
            if strategy == "random":
                return random.sample(jpg_files, sample_size)
            elif strategy == "first":
                return jpg_files[:sample_size]
            elif strategy == "last":
                return jpg_files[-sample_size:]
            elif strategy == "middle":
                mid = len(jpg_files) // 2
                half_size = sample_size // 2
                return jpg_files[mid - half_size:mid + half_size]
            else:
                raise ValueError(f"Unknown sample strategy: {strategy}")
        return jpg_files



class BatchProcessor:
    """Coordinates the overall batch processing workflow."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.src_dir, self.output_dir = self.setup_paths()

    def setup_paths(self):
        batch_id = self.cfg.batch_id

        primary_storage_developed = Path(self.cfg.paths.primary_storage, "semifield-developed-images")
        secondary_storage_developed = Path(self.cfg.paths.secondary_storage, "semifield-developed-images")
        tertiary_storage_developed = Path(self.cfg.paths.tertiary_storage, "semifield-developed-images")

        src_dir = primary_storage_developed / batch_id / "images"
        if not src_dir.exists():
            src_dir = secondary_storage_developed / batch_id / "images"
            if not src_dir.exists():
                src_dir = tertiary_storage_developed / batch_id / "images"
                if not src_dir.exists():
                    log.error(f"Source directory {src_dir} does not exist after checking all storage locations. Exiting.")
                    exit(1)
            
        if "longterm_images2" in str(src_dir):
            lts_name = "longterm_images2"
        elif "GROW_DATA" in str(src_dir):
            lts_name = "GROW_DATA"
        else:
            lts_name = "longterm_images"
        
        output_dir = Path(self.cfg.paths.data_dir) / lts_name / "semifield-developed-images" / batch_id / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        return src_dir, output_dir


    def copy_files(self):
        jpg_files = FileManager.get_sampled_files(
            self.src_dir, self.cfg.inspect_developed.sample_size, self.cfg.inspect_developed.sample_strategy
        )
        FileManager.copy_files_in_parallel(self.src_dir, self.output_dir, jpg_files)

    def downscale_images(self):
        local_jpgs = list(self.output_dir.glob("*.jpg"))
        if not local_jpgs:
            log.info(f"No JPG files found in {self.output_dir}")
            return
        downscaled_output_dir = Path(self.output_dir.parent, "downscaled")
        downscaled_output_dir.mkdir(parents=True, exist_ok=True)
        downscale_factor = self.cfg.inspect_developed.downscale_factor
        
        if downscale_factor == 1:
            log.info("Skipping downscaling as factor is 1")
            return
        
        if downscale_factor is None:
            log.error("Skipping downscaling as factor is not provided")
            return
        
        for image_path in sorted(local_jpgs):
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            new_height = int(height * downscale_factor)
            new_width = int(width * downscale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            output_image_path = downscaled_output_dir / image_path.name
            cv2.imwrite(str(output_image_path), resized_image)
            log.info(f"Saved: {output_image_path}")
        
    
    def remove_local_src_images(self):
        # Remove image from source directory
        remove_jpgs = self.cfg.inspect_developed.remove_jpgs
        downscale_factor = self.cfg.inspect_developed.downscale_factor
        if remove_jpgs and (downscale_factor != 1 or downscale_factor is not None):
            local_jpgs = list(self.output_dir.glob("*.jpg"))
            for image_path in local_jpgs:
                image_path.unlink()
                log.info(f"Removed JPG: {image_path}")

    def run(self):
        log.info(f"Processing batch {self.cfg.batch_id}")
        self.copy_files()
        log.info(f"Downscaling images in {self.output_dir}")
        self.downscale_images()
        log.info(f"Removing source images in {self.output_dir}")
        self.remove_local_src_images()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    setup_signal_handler()
    batch_processor = BatchProcessor(cfg)
    batch_processor.run()
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
