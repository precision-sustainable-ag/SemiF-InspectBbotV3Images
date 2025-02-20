import numpy as np
import cv2
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import os
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import random

log = logging.getLogger(__name__)


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
    def get_sampled_files(raw_files: list, sample_size: int, strategy: str = "random") -> list:
        """Samples a subset of files from a directory based on a given strategy."""
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
        # source_b, source_g, source_r = cv2.split(source_flt)
        source_r, source_g, source_b = cv2.split(source_flt)
        
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
    def demosaic_image(raw_file: Path, cfg: DictConfig):
        """Demosaics a RAW image file using bilinear interpolation."""
        log.info(f"Demosaicing: {raw_file}")
        im_height, im_width = cfg.inspect_v31.image_height, cfg.inspect_v31.image_width

        nparray = np.fromfile(raw_file, dtype=np.uint16).reshape((im_height, im_width))
        image_data = nparray.astype(np.float32) / 65535.0

        demosaiced = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
        demosaiced = np.clip(demosaiced, 0, 1)
        return demosaiced

    @staticmethod
    def resize_image(image: np.ndarray, downscale_factor: float):
        """Downscales an image file by a given factor."""
        height, width = image.shape[:2]
        new_height = int(height * downscale_factor)
        new_width = int(width * downscale_factor)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Path):
        """Saves an image to disk."""
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        log.info(f"Saved: {output_path}")
    
    @staticmethod
    def remove_local_raw(local_raw_path: Path):
        """Removes an image file from disk."""
        # Sanity check
        if "research-project" in str(local_raw_path) or "screberg" in str(local_raw_path):
            log.warning("Refusing to remove file from LTS research-project directory.")
            return
        local_raw_path.unlink()
        log.info(f"Removed raw image: {local_raw_path}")

    @staticmethod
    def process_image(raw_file: Path, cfg: DictConfig, transformation_matrix, output_dir: Path, local_raw_dir: Path):
        log.info(f"Processing: {raw_file}")

        # Copy LTS raw to local raw
        local_raw_path = local_raw_dir / raw_file.name
        FileManager.copy_file(raw_file, local_raw_path)

        # Demosaic
        demosaiced_rgb = ImageProcessor.demosaic_image(local_raw_path, cfg)

        # Apply color correction
        corrected_image_rgb = ImageProcessor.apply_transformation_matrix(demosaiced_rgb, transformation_matrix)

        # Convert to 8-bit or 16-bit RGB
        bit_depth = cfg.inspect_v31.bit_depth
        assert bit_depth in [8, 16], f"Unsupported bit depth: {bit_depth}"
        rgb_bit_image = (corrected_image_rgb * 255).astype(np.uint8) if bit_depth == 8 else (corrected_image_rgb * 65535).astype(np.uint16)

        # Covert to BGR
        bgr_bit_image = cv2.cvtColor(rgb_bit_image, cv2.COLOR_RGB2BGR)

        # Downscale if necessary
        downscale_factor = cfg.inspect_v31.downscale.factor
        if downscale_factor != 1.0:
            bgr_bit_image = ImageProcessor.resize_image(bgr_bit_image, downscale_factor)
        
        # Save the final image
        ImageProcessor.save_image(bgr_bit_image, output_dir / f"{local_raw_path.stem}.jpg")

        # Remove local raw file
        if cfg.inspect_v31.downscale.remove_local_raws:
            local_raw_path = local_raw_dir / raw_file.name
            ImageProcessor.remove_local_raw(local_raw_path)

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
        self.src_dir, self.local_raw_dir, self.output_dir = self.setup_paths()

    def setup_paths(self):
        """Sets up and validates required directories for processing.
        
        Returns:
            tuple: Paths to source, raw, output, and downscaled directories.
        """
        src_dir = Path(self.cfg.paths.primary_storage, "semifield-upload", self.batch_id)
        local_raw_dir = Path(self.cfg.paths.local_upload) / self.batch_id / "raw"
        output_dir = Path(self.cfg.paths.local_upload) / self.batch_id / "colorcorrected"
        
        if self.downscale_factor != 1.0:
            output_dir = Path(self.cfg.paths.local_upload) / self.batch_id / "downscaled_colorcorrected"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        local_raw_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")
        return src_dir, local_raw_dir, output_dir

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

    def process_files(self, transformation_matrix):
        """Processes RAW image files by applying demosaicing and color correction.
        
        Args:
            transformation_matrix (np.ndarray): Transformation matrix for color correction.
        """
        all_raw_files = list(self.src_dir.glob("*.RAW"))
        log.info(f"Found {len(all_raw_files)} RAW files.")
        
        # Determine the largest file size to filter out incomplete or corrupted files
        max_file_size = max(f.stat().st_size for f in all_raw_files)
        raw_files = [f for f in all_raw_files if f.stat().st_size == max_file_size]
        
        raw_files = FileManager.get_sampled_files(
            raw_files, self.cfg.inspect_v31.sample_size, self.cfg.inspect_v31.sample_strategy
        )

        if not raw_files:
            log.info("No new files to process.")
            return
        
        log.info(f"Processing {len(raw_files)} RAW files.")
        
        with ProcessPoolExecutor(max_workers=self.cfg.inspect_v31.concurrent_workers) as executor:
            futures = [
                executor.submit(ImageProcessor.process_image, raw_file, self.cfg, transformation_matrix, self.output_dir, self.local_raw_dir)
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
        """Executes the full batch processing workflow."""
        transformation_matrix = self.load_transformation_matrix()
        self.process_files(transformation_matrix)
        
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for batch image processing."""
    random.seed(cfg.inspect_v31.random_seed)
    batch_processor = BatchProcessor(cfg)
    batch_processor.run()
    log.info("Batch processing completed.")


if __name__ == "__main__":
    main()
