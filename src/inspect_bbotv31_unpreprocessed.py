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
        """Copies a file to the destination if it doesn't already exist or has a different size."""
        if not os.path.exists(dest) or os.path.getsize(src) != os.path.getsize(dest):
            shutil.copy2(src, dest)
            log.info(f"Copied {src} to {dest}")
        else:
            log.info(f"Skipped {src}, already present at destination with matching size")

    @staticmethod
    def copy_files_in_parallel(src_dir: Path, dest_dir: Path, files: list, max_workers=12):
        """Copies multiple files in parallel using threading."""
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
        """Samples a subset of files from a directory based on a given strategy."""
        raw_files = sorted(list(raw_dir.glob("*.RAW")))
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

        # Extract color channel coefficients from transformation matrix
        red, green, blue, *_ = np.split(transformation_matrix, 9, axis=1)

        # Normalize the source image to the range [0, 1]
        source_dtype = source_img.dtype
        max_val = np.iinfo(source_dtype).max if source_dtype.kind == 'u' else 1.0
        source_flt = source_img.astype(np.float64) / max_val
        source_b, source_g, source_r = cv2.split(source_flt)
        
        # Compute powers of source image
        source_b2, source_b3 = source_b**2, source_b**3
        source_g2, source_g3 = source_g**2, source_g**3
        source_r2, source_r3 = source_r**2, source_r**3
        
        # Compute color transformation
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
    """Coordinates the overall batch processing workflow, including file management,
    image processing, and downscaling."""

    def __init__(self, cfg: DictConfig):
        """Initializes the batch processor with configuration settings.
        
        Args:
            cfg (DictConfig): Configuration object containing paths and processing parameters.
        """
        self.cfg = cfg
        self.batch_id = cfg.batch_id
        self.downscale_factor = cfg.inspect_v31.downscale.factor
        self.remove_images = cfg.inspect_v31.downscale.remove_images
        self.remove_raws = cfg.inspect_v31.downscale.remove_raws
        self.src_dir, self.raw_dir, self.output_dir, self.downscaled_dir = self.setup_paths()

    def setup_paths(self):
        """Sets up and validates required directories for processing.
        
        Returns:
            tuple: Paths to source, raw, output, and downscaled directories.
        """
        self.src_dir = Path(self.cfg.paths.primary_storage, "semifield-upload", self.batch_id)
        self.raw_dir = Path(self.cfg.paths.local_upload) / self.batch_id / "raw"
        self.output_dir = Path(self.cfg.paths.local_upload) / self.batch_id / "colorcorrected"
        self.downscaled_dir = Path(self.cfg.paths.local_upload) / self.batch_id / "downscaled"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downscaled_dir.mkdir(parents=True, exist_ok=True)

        if not self.src_dir.exists():
            raise FileNotFoundError(f"Source directory {self.src_dir} does not exist.")
        return self.src_dir, self.raw_dir, self.output_dir, self.downscaled_dir

    def load_transformation_matrix(self) -> np.ndarray:
        """Loads the color transformation matrix from file.
        
        Returns:
            np.ndarray: The loaded transformation matrix.
        """
        color_matrix_path = Path(self.cfg.paths.color_matrix)
        if not color_matrix_path.exists():
            raise FileNotFoundError(f"Color matrix file {color_matrix_path} not found.")
        with np.load(color_matrix_path) as data:
            return data["matrix"]

    def copy_files(self):
        """Copies a sample of RAW files from the source directory to the local raw directory."""
        raw_files = FileManager.get_sampled_files(
            self.src_dir, self.cfg.inspect_v31.sample_size, self.cfg.inspect_v31.sample_strategy
        )
        FileManager.copy_files_in_parallel(self.src_dir, self.raw_dir, raw_files)

    def process_files(self, transformation_matrix):
        """Processes RAW image files by applying demosaicing and color correction.
        
        Args:
            transformation_matrix (np.ndarray): Transformation matrix for color correction.
        """
        all_raw_files = list(self.raw_dir.glob("*.RAW"))
        log.info(f"Found {len(all_raw_files)} RAW files.")
        
        # Determine the largest file size to filter out incomplete or corrupted files
        max_file_size = max(f.stat().st_size for f in all_raw_files)
        raw_files = [f for f in all_raw_files if f.stat().st_size == max_file_size]
        
        # Filter out already processed files
        raw_files = [f for f in raw_files if not (self.output_dir / f"{f.stem}.jpg").exists()]
        
        if not raw_files:
            log.info("No new files to process.")
            return
        
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

    def downscale_images(self):
        """Downscales processed images if required and removes original files if configured."""
        input_images = sorted(list(self.output_dir.glob("*.jpg")))
        
        for image_path in input_images:
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            new_height = int(height * self.downscale_factor)
            new_width = int(width * self.downscale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            output_image_path = self.downscaled_dir / image_path.name
            cv2.imwrite(str(output_image_path), resized_image)
            log.info(f"Saved: {output_image_path}")
            
            # Remove original processed images if configured
            if self.remove_images:
                image_path.unlink()
                log.info(f"Removed JPG: {image_path}")
            
            # Remove original RAW files if configured
            if self.remove_raws:
                raw_file = self.raw_dir / f"{image_path.stem}.RAW"
                if raw_file.exists():
                    raw_file.unlink()
                    log.info(f"Removed RAW: {raw_file}")
    
    def run(self):
        """Executes the full batch processing workflow."""
        self.copy_files()
        transformation_matrix = self.load_transformation_matrix()
        self.process_files(transformation_matrix)
        self.downscale_images()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for batch image processing."""
    random.seed(cfg.inspect_v31.random_seed)
    setup_signal_handler()
    batch_processor = BatchProcessor(cfg)
    batch_processor.run()
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
