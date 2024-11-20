import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import re
from utils.utils import (
    get_size, read_jpg, read_raw, convert_epoch_to_edt, get_exif
)

log = logging.getLogger(__name__)


class ImageBatchProcessor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.batch_par = Path(config.paths.primary_storage)
        self.report_dir = Path(config.paths.reports, "batch_inspection")
        self.ext = config.ext
        self.sample_size = config.sample_size
        self.sample_strategy = config.sample_strategy
        self.rescale_factor = config.plotting_rescale_factor
        self.batch_file = Path(config.paths.batches_in_storage)
        self.state_prefix = config.filter_by_state

    @staticmethod
    def ensure_directory(path):
        """Ensure that the directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_empty_folder(folder):
        """Check if a folder is empty."""
        return not any(folder.iterdir())

    @staticmethod
    def get_sample(images, sample_size, strategy):
        """Select a sample of images based on the given strategy."""
        if strategy == "random":
            return random.sample(images, sample_size)
        elif strategy == "first":
            return images[:sample_size]
        elif strategy == "last":
            return images[-sample_size:]
        elif strategy == "middle":
            mid = len(images) // 2
            half_size = sample_size // 2
            return images[mid - half_size:mid + half_size]
        else:
            raise ValueError(f"Unknown sample strategy: {strategy}")
        
    def is_batch_inspected(self, batch_name):
        """Check if the batch has already been inspected."""
        batch_report_dir = self.report_dir / batch_name
        if not batch_report_dir.exists():
            return False

        inspected_files = list(batch_report_dir.glob("*.png"))
        return len(inspected_files) > 0

    def plot_image_capture_times(self, batch_name, image_name=None):
        """Plot image capture times for a batch."""

        batch_dir = self.batch_par / batch_name

        if self.check_empty_folder(batch_dir):
            print(f"Skipping {batch_name}: Folder is empty.")
            return

        imgs = [str(x.name) for x in batch_dir.glob(f"*.{self.ext}")]
        total_images = len(imgs)

        if total_images == 0:
            print(f"Skipping {batch_name}: No {self.ext} files found.")
            return

        try:
            epoch_times = [int(name.split('_')[1].split('.')[0]) for name in imgs]
        except Exception as e:
            return
        epoch_times.sort()
        dates = [datetime.fromtimestamp(epoch) for epoch in epoch_times]

        time_diffs = [epoch_times[i+1] - epoch_times[i] for i in range(len(epoch_times) - 1)]
        avg_time_diff_seconds = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        avg_time_diff_minutes = avg_time_diff_seconds / 60

        df = pd.DataFrame({'Date': dates})
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], range(len(df)), marker='o', linestyle='-', color='b')

        if image_name:
            try:
                image_epoch = int(image_name.split('_')[1].split('.')[0])
                image_date = datetime.fromtimestamp(image_epoch)
                image_index = dates.index(image_date)
                plt.scatter([image_date], [image_index], color='r', zorder=5)
                plt.annotate(image_name, (image_date, image_index), textcoords="offset points", xytext=(0, 10), ha='center')
            except ValueError:
                print(f"Provided image name {image_name} does not exist in the batch.")

        plt.xlabel('Capture Time (EDT)')
        plt.ylabel('Image Index')
        plt.title(f'Image Capture Times: {batch_name}')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()

        plot_path = self.report_dir / batch_name
        self.ensure_directory(plot_path)
        plot_filename = plot_path / f"{batch_name}_capture_time_plot.png"
        plt.savefig(plot_filename)
        plt.show()
        print(f"Saved plot: {plot_filename}")

        total_time_seconds = epoch_times[-1] - epoch_times[0] if epoch_times else 0
        total_time = timedelta(seconds=total_time_seconds)
        print(f"Average time between captures: {avg_time_diff_seconds:.2f} sec ({avg_time_diff_minutes:.2f} min)")
        print(f"Total time to collect images: {total_time}")
        print(f"Total number of raw images: {total_images}")

    def inspect_images(self, batch_name):
        """Inspect images for a batch."""
        
        batchdir = self.batch_par / batch_name

        if self.check_empty_folder(batchdir):
            print(f"Skipping {batch_name}: Folder is empty.")
            return

        imgs_to_show = sorted([x for x in batchdir.glob(f"*.{self.ext}")])
        reports_path = self.report_dir / batch_name
        self.ensure_directory(reports_path)

        plt.close("all")
        for img in tqdm(self.get_sample(imgs_to_show, self.sample_size, self.sample_strategy)):
            imgsize_str, imgsize = get_size(img)
            exif = get_exif(img)
            if self.ext == "ARW" and imgsize < 100:
                print(f"Skipping {img}: File size {imgsize} too small.")
                continue

            print(f"Processing {img} with size {imgsize}.")
            try:
                im = read_raw(str(img)) if self.ext == "ARW" else read_jpg(img)
            except Exception as e:
                print(f"Error reading {img}: {e}")
                continue
            resized = cv2.resize(im, (int(im.shape[1] * self.rescale_factor), int(im.shape[0] * self.rescale_factor)))
            plt.imshow(resized)

            image_name = img.name
            epoch_time = convert_epoch_to_edt(int(img.stem.split("_")[-1]))

            plt.title(f"{image_name} ({epoch_time})")
            text_info = (
                f'FNumber: {exif["FNumber"]}\n'
                f'ISO: {exif["ISOSpeedRatings"]}\n'
                f'Exposure: {exif["ExposureTime"]}'
            )
            plt.text(
                0.95, 0.95, text_info, horizontalalignment='right',
                verticalalignment='top', transform=plt.gca().transAxes,
                bbox=dict(facecolor='lightblue', alpha=0.5), fontsize=8
            )
            plt.tight_layout()

            plot_filename = reports_path / f"{img.stem}.png"
            plt.savefig(plot_filename)
            print(f"Saved plot: {plot_filename}")
            plt.close()

    def process_batches(self):
        """Process batches listed in a file."""

        with open(self.batch_file, "r") as f:
            batches = [line.strip() for line in f if line.strip() and self.is_valid_format(line.strip())]
            if self.state_prefix:
                batches = [line for line in batches if line.split("_")[0] == self.state_suffix]


        for batch_name in batches:
            if self.is_batch_inspected(batch_name):
                print(f"Skipping {batch_name}: Batch has already been inspected.")
                continue
        
            print(f"Processing batch: {batch_name}")
            self.plot_image_capture_times(batch_name)
            self.inspect_images(batch_name)

    @staticmethod
    def is_valid_format(s):
        """Validate batch name format."""
        pattern = r'^[A-Z]{2}_\d{4}-\d{2}-\d{2}$'
        return bool(re.match(pattern, s))


def main(cfg: DictConfig) -> None:
    processor = ImageBatchProcessor(cfg)
    processor.process_batches()
