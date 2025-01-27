import numpy as np
import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
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
    def copy_files_in_parallel(src_dir: Path, dest_dir: Path, files: list, max_workers=12):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
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
        raw_files = list(raw_dir.glob("*.RAW"))
        if sample_size and len(raw_files) > sample_size:
            if strategy == "random":
                return random.sample(raw_files, sample_size)
            elif strategy == "first":
                return raw_files[:sample_size]
            elif strategy == "last":
                return raw_files[-sample_size:]
            elif strategy == "middle":
                mid = len(raw_files) // 2
                half_size = sample_size // 2
                return raw_files[mid - half_size:mid + half_size]
            else:
                raise ValueError(f"Unknown sample strategy: {strategy}")
        return raw_files


class ImageProcessor:
    """Handles image processing tasks like demosaicing and color correction."""

    @staticmethod
    def apply_transformation_matrix(source_img: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """Apply a transformation matrix to the source image to correct its color space."""
        if transformation_matrix.shape != (9, 9):
            log.error("Transformation matrix must be a 9x9 matrix.")
            return None

        if source_img.ndim != 3:
            log.error("Source image must be an RGB image.")
            return None

        red, green, blue, *_ = np.split(transformation_matrix, 9, axis=1)

        source_dtype = source_img.dtype
        max_val = np.iinfo(source_dtype).max if source_dtype.kind == 'u' else 1.0

        source_flt = source_img.astype(np.float64) / max_val
        source_b, source_g, source_r = cv2.split(source_flt)

        source_b2, source_b3 = source_b**2, source_b**3
        source_g2, source_g3 = source_g**2, source_g**3
        source_r2, source_r3 = source_r**2, source_r**3

        b = (source_r * blue[0] + source_g * blue[1] + source_b * blue[2] +
            source_r2 * blue[3] + source_g2 * blue[4] + source_b2 * blue[5] +
            source_r3 * blue[6] + source_g3 * blue[7] + source_b3 * blue[8])
        
        g = (source_r * green[0] + source_g * green[1] + source_b * green[2] +
            source_r2 * green[3] + source_g2 * green[4] + source_b2 * green[5] +
            source_r3 * green[6] + source_g3 * green[7] + source_b3 * green[8])
        
        r = (source_r * red[0] + source_g * red[1] + source_b * red[2] +
            source_r2 * red[3] + source_g2 * red[4] + source_b2 * red[5] +
            source_r3 * red[6] + source_g3 * red[7] + source_b3 * red[8])

        corrected_img = cv2.merge([r, g, b])
        corrected_img = np.clip(corrected_img * max_val, 0, max_val).astype(source_dtype)
        return corrected_img

    @staticmethod
    def process_image(raw_file: Path, cfg: DictConfig, transformation_matrix: np.ndarray, output_dir: Path):
        log.info(f"Processing: {raw_file}")
        im_height, im_width = cfg.inspect_v31.image_height, cfg.inspect_v31.image_width
        bit_depth = cfg.inspect_v31.bit_depth

        nparray = np.fromfile(raw_file, dtype=np.uint16).reshape((im_height, im_width))
        image_data = nparray.astype(np.float32) / 65535.0

        demosaiced = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
        demosaiced = np.clip(demosaiced, 0, 1)
        corrected_image = ImageProcessor.apply_transformation_matrix(demosaiced, transformation_matrix)

        output_image = (corrected_image * 255 if bit_depth == 8 else corrected_image * 65535).astype(np.uint8)
        output_file = output_dir / f"{raw_file.stem}.jpg"
        cv2.imwrite(str(output_file), output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        log.info(f"Saved: {output_file}")


class BatchProcessor:
    """Coordinates the overall batch processing workflow."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.src_dir, self.raw_dir, self.output_dir = self.setup_paths()

    def setup_paths(self):
        batch_id = self.cfg.batch_id
        src_dir = Path(self.cfg.paths.primary_storage_uploads) / batch_id
        raw_dir = Path(self.cfg.paths.local_upload) / batch_id / "raw"
        output_dir = Path(self.cfg.paths.local_upload) / batch_id / "colorcorrected"

        raw_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
        return src_dir, raw_dir, output_dir

    def load_transformation_matrix(self) -> np.ndarray:
        color_matrix_path = Path(self.cfg.paths.color_matrix)
        if not color_matrix_path.exists():
            raise FileNotFoundError(f"Color matrix file {color_matrix_path} not found.")
        with np.load(color_matrix_path) as data:
            return data["matrix"]

    def copy_files(self):
        raw_files = FileManager.get_sampled_files(
            self.src_dir, self.cfg.inspect_v31.sample_size, self.cfg.inspect_v31.sample_strategy
        )
        FileManager.copy_files_in_parallel(self.src_dir, self.raw_dir, raw_files)

    def process_files(self, transformation_matrix):
        all_raw_files = list(self.raw_dir.glob("*.RAW"))
        log.info(f"Found {len(all_raw_files)} RAW files.")
        # Filter out any files that have a smaller size than the largest file size in the directory
        max_file_size = max(f.stat().st_size for f in all_raw_files)
        raw_files = [f for f in all_raw_files if f.stat().st_size == max_file_size]
        # Filter out any files that have already been processed
        raw_files = [f for f in raw_files if not (self.output_dir / f"{f.stem}.jpg").exists()]
        
        if not raw_files:
            log.info("No new files to process.")
            return
        
        elif len(raw_files) < len(all_raw_files):
            log.info(f"Filtered out {len(all_raw_files) - len(raw_files)} already processed files.")
        
        else:
            log.info(f"Processing {len(raw_files)} RAW files.")
        
        with ProcessPoolExecutor(max_workers=self.cfg.inspect_v31.concurrent_workers) as executor:
            futures = [
                executor.submit(ImageProcessor.process_image, raw_file, self.cfg, transformation_matrix, self.output_dir)
                for raw_file in raw_files
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except ValueError as e:
                    log.error(f"Error processing file: {e}")
                except KeyboardInterrupt:
                    log.info("Batch processing interrupted.")


    def run(self):
        # self.copy_files()
        transformation_matrix = self.load_transformation_matrix()
        self.process_files(transformation_matrix)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    setup_signal_handler()
    batch_processor = BatchProcessor(cfg)
    batch_processor.run()
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
