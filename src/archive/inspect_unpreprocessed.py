import cv2
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm
import logging
import hydra
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear
import random
import shutil
import signal
from signal import SIGINT

log = logging.getLogger(__name__)

def terminate_executor(executor):
    """Gracefully terminate executor by shutting down and waiting."""
    executor.shutdown(wait=True, cancel_futures=True)
    log.info("Executor terminated.")

# Add a signal handler to stop child processes
def signal_handler(signal_received, frame):
    log.info("Interrupt signal received. Cleaning up...")
    raise KeyboardInterrupt

# Register signal handler
signal.signal(SIGINT, signal_handler)

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

    corrected_img = cv2.merge([b, g, r])
    corrected_img = np.clip(corrected_img * max_val, 0, max_val).astype(source_dtype)
    return corrected_img

class RawFileHandler:
    def __init__(self, src_dir: Path, dest_dir: Path, selection_mode: str, sample_number: int):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.selection_mode = selection_mode
        self.sample_number = sample_number
        log.info("RawFileHandler initialized with source directory: %s and destination directory: %s", self.src_dir, self.dest_dir)


    def select_raw_files(self, raw_files):
        log.info("Selecting raw files using mode: %s and sample number: %d", self.selection_mode, self.sample_number)
        if self.selection_mode == "first":
            return raw_files[:self.sample_number]
        elif self.selection_mode == "last":
            return raw_files[-self.sample_number:]
        elif self.selection_mode == "random":
            return random.sample(raw_files, min(self.sample_number, len(raw_files)))
        return []

    def copy_files(self, max_workers=12, raw_extension=".RAW"):
        log.info("Copying raw files with extension %s from %s to %s", raw_extension, self.src_dir, self.dest_dir)   
        self.dest_dir.mkdir(parents=True, exist_ok=True)

        raw_files = list(self.src_dir.glob(f"*{raw_extension}"))
        raw_files = self.select_raw_files(raw_files)
        
        existing_files = list(self.dest_dir.glob(f"*{raw_extension}"))
        existing_file_stems = {file.stem for file in self.dest_dir.glob(f"*{raw_extension}")}
        filtered_files = [file for file in raw_files if file.stem not in existing_file_stems]
        log.info("Filtered %d files to copy.", len(filtered_files))
        if not filtered_files:
            log.info(f"No files to copy. {len(existing_files)} files already present at destination.")
            return existing_files
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._copy_file, file, self.dest_dir / file.name)
                for file in filtered_files
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error copying file: {e}")
        filtered_files = [self.dest_dir / file.name for file in filtered_files]
        return filtered_files

    @staticmethod
    def _copy_file(src: Path, dest: Path):
        if not dest.exists() or src.stat().st_size != dest.stat().st_size:
            shutil.copy2(src, dest)
            log.info(f"Copied {src} to {dest}")
        else:
            log.info(f"Skipped {src}, already present at destination with matching size")


class DemosaicProcessor:
    def __init__(self, im_height: int, im_width: int, bit_depth: int):
        self.im_height = im_height
        self.im_width = im_width
        self.bit_depth = bit_depth
        log.info("DemosaicProcessor initialized with height: %d, width: %d, bit depth: %d", self.im_height, self.im_width, self.bit_depth)


    def process(self, raw_files, output_dir: Path, max_workers: int, concurrent_processing=True):
        log.info("Starting demosaicing process with %d files.", len(raw_files))

        output_dir.mkdir(parents=True, exist_ok=True)
        if concurrent_processing:
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._process_image, file, output_dir)
                    for file in raw_files
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"Error processing image: {e}")
        else:
            for file in raw_files:
                self._process_image(file, output_dir)

    def _process_image(self, raw_file: Path, output_dir: Path):
        log.debug("Processing file: %s", raw_file)

        try:
            nparray = np.fromfile(raw_file, dtype=np.uint16).astype(np.uint16)

            org_reshaped = nparray.reshape((self.im_height, self.im_width))
            image_data = org_reshaped.astype(np.float32) / 65535.0

            rgb_demosaiced = demosaicing_CFA_Bayer_bilinear(image_data, pattern="RGGB")
            rgb_demosaiced_adjusted = np.clip(rgb_demosaiced, 0, 1)

            if self.bit_depth == 8:
                rgb_colour_image = (rgb_demosaiced_adjusted * 255).astype(np.uint8)
            else:
                rgb_colour_image = (rgb_demosaiced_adjusted * 65535).astype(np.uint16)

            bgr_colour_image = cv2.cvtColor(rgb_colour_image, cv2.COLOR_RGB2BGR)
            output_file = output_dir / f"{raw_file.stem}.png"
            cv2.imwrite(str(output_file), bgr_colour_image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            log.info(f"Saved image to {output_file} with {self.bit_depth}-bit depth")
        except Exception as e:
            log.error(f"Failed to process image {raw_file}: {e}")


class ColorCorrectionProcessor:
    def __init__(self, transformation_matrix: np.ndarray):
        self.transformation_matrix = transformation_matrix
        log.info("ColorCorrectionProcessor initialized.")

    def process(self, images, output_dir: Path, downscale_factor=None, max_workers=8, concurrent_processing=True):
        """Process images concurrently using concurrent.futures."""
        log.info("Starting color correction process for %d images.", len(images))

        output_dir.mkdir(parents=True, exist_ok=True)

        if concurrent_processing:
            log.info(f"Processing images concurrently with {max_workers} workers.")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit each image for concurrent processing
                futures = {
                    executor.submit(self._apply_correction, img_path, output_dir, downscale_factor): img_path 
                    for img_path in images
                }

                for future in tqdm(as_completed(futures), total=len(images)):
                    img_path = futures[future]
                    try:
                        future.result()  # This raises any exception that occurred
                    except Exception as e:
                        log.exception(f"Error processing {img_path}: {e}")
        else:
            log.info("Processing images sequentially.")
            for img_path in tqdm(images):
                try:
                    log.info("Applying color correction to image: %s", img_path)
                    self._apply_correction(img_path, output_dir, downscale_factor)
                except Exception as e:
                    log.exception(f"Error processing {img_path}: {e}")

    def _apply_correction(self, img_path: Path, output_dir: Path, downscale_factor):
        """Applies the color correction to an individual image."""
        log.debug("Reading image: %s", img_path)        
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            corrected_img = apply_transformation_matrix(img, self.transformation_matrix)
            corrected_img = (corrected_img * 255).astype(np.uint8)

            if downscale_factor:
                log.info("Downscaling image: %s", img_path)
                corrected_img = cv2.resize(corrected_img, (0, 0), fx=downscale_factor, fy=downscale_factor)

            output_file = output_dir / f"{img_path.stem}.png"
            # cv2.imwrite(str(output_file), cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(str(output_file), cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
            log.info(f"Saved corrected image to {output_file}")
        except Exception as e:
            log.exception(f"Error in _apply_correction for {img_path}: {e}")
            raise


class PipelineProcessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.max_workers = cfg.inspect_unpreprocessed.max_workers
        self.demosaic_concurrently = cfg.inspect_unpreprocessed.demosaic_concurrently
        self.ccm_concurrently = cfg.inspect_unpreprocessed.ccm_concurrently
        self.scale_factor = cfg.inspect_unpreprocessed.scale_factor
        self.remove_raws = cfg.inspect_unpreprocessed.remove_local_raws
        self.remove_demosaiced = cfg.inspect_unpreprocessed.remove_demosaiced
        self.sample_strategy = cfg.inspect_unpreprocessed.sample_strategy 
        self.sample_size = cfg.inspect_unpreprocessed.sample_size
        log.info("PipelineProcessor initialized.")

    def run(self):
        batch_id = self.cfg.inspect_unpreprocessed.batch_id
        longterm_raw_dir = Path(self.cfg.paths.primary_storage_uploads, batch_id)
        temp_raw_dir = Path(self.cfg.paths.temp_data, batch_id, "raws")
        temp_demosaiced_dir = Path(self.cfg.paths.temp_data, batch_id, "demosaiced")
        temp_color_corrected_dir = Path(self.cfg.paths.temp_data, batch_id, "color_corrected")
        log.info("Running pipeline for batch ID: %s", batch_id)

        assert longterm_raw_dir.exists(), f"Source directory {longterm_raw_dir} does not exist."

        local_raw_files = list(temp_raw_dir.glob("*.RAW"))
        if self.cfg.inspect_unpreprocessed.pipeline.copy_from_lockers:
            log.info("Starting raw file handling process.")

            handler = RawFileHandler(longterm_raw_dir, temp_raw_dir, selection_mode=self.sample_strategy, sample_number=self.sample_size)
            local_raw_files = handler.copy_files()

        if self.cfg.inspect_unpreprocessed.pipeline.demosaic:
            log.info("Starting demosaicing process.")

            processor = DemosaicProcessor(
                im_height=self.cfg.inspect_unpreprocessed.im_height,
                im_width=self.cfg.inspect_unpreprocessed.im_width,
                bit_depth=self.cfg.inspect_unpreprocessed.bit_depth,
            )
            assert len(local_raw_files) != 0, "No raw files to process."
            processor.process(local_raw_files, temp_demosaiced_dir, max_workers=self.max_workers, concurrent_processing=self.demosaic_concurrently)

        
        if self.cfg.inspect_unpreprocessed.pipeline.color_correct:
            log.info("Starting color correction process.")
            
            ccm_matrix = np.load(self.cfg.paths.color_matrix)['matrix']
            processor = ColorCorrectionProcessor(transformation_matrix=ccm_matrix)
            if len(local_raw_files) == 0:
                local_raw_files = list(temp_demosaiced_dir.glob("*.png"))
                local_raw_files = random.sample(local_raw_files, min(self.sample_size, len(local_raw_files)))
            images = [Path(temp_demosaiced_dir, file.stem + ".png") for file in local_raw_files]
            processor.process(images, temp_color_corrected_dir, downscale_factor=self.scale_factor, max_workers=self.max_workers, concurrent_processing=True),

        if self.remove_demosaiced:
            demosaic_files = list(temp_demosaiced_dir.glob("*.png"))
            log.info("Removing demosaiced files.")
            for file in demosaic_files:
                file.unlink()
        
        if self.remove_raws:
            log.info("Removing local raw files.")
            local_raw_files = list(temp_raw_dir.glob("*.RAW"))
            for file in local_raw_files:
                file.unlink()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    pipeline = PipelineProcessor(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
