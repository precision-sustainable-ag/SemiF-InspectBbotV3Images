import hydra
from omegaconf import DictConfig
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import random

log = logging.getLogger(__name__)

class ImageReport:
    def __init__(self, cfg: DictConfig):
        self.batch_id = cfg.batch_id
        self.output_report_dir = Path(cfg.paths.local_upload) / self.batch_id / "report"
        self.output_report_dir.mkdir(parents=True, exist_ok=True)

        self.plot_file_base = self.output_report_dir / "plots"
        self.plot_file_base.mkdir(parents=True, exist_ok=True)

        self.csv_output_dir = self.output_report_dir / "data"
        self.csv_output_dir.mkdir(parents=True, exist_ok=True)

        self.upload_directory = Path(cfg.paths.primary_storage, "semifield-upload") / self.batch_id
        self.image_files = list(self.upload_directory.glob("*.RAW"))  # Adjust extension if necessary
        self.image_data = []

        self.local_sample_dir = Path(cfg.paths.local_upload) / self.batch_id / "sample_images"

    def calculate_total_images(self):
        return len(self.image_files)

    def calculate_total_size(self):
        return sum(image.stat().st_size for image in self.image_files)

    def calculate_average_size(self):
        total_images = self.calculate_total_images()
        total_size = self.calculate_total_size()
        return total_size / total_images if total_images > 0 else 0
    
    def find_max_image_size(self):
        return max((image.stat().st_size for image in self.image_files), default=0)

    def count_partial_uploads(self):
        max_size = self.find_max_image_size()
        return sum(1 for image in self.image_files if image.stat().st_size < max_size)

    def extract_image_metadata(self):
        for image in self.image_files:
            name = image.stem
            state, epoch = name.split("_")
            file_size = image.stat().st_size
            file_mtime = datetime.fromtimestamp(image.stat().st_mtime)
            capture_datetime = datetime.fromtimestamp(int(epoch))
            average_upload_time = file_mtime - capture_datetime
            
            data = {
                "batch_id": self.batch_id,
                "filename": image.name,
                "state": state,
                "epoch": int(epoch),
                "file_size_bytes": file_size,
                "file_size_kib": file_size / 1024,
                "file_datetime_est_modified": file_mtime, 
                "capture_datetime_epoch": capture_datetime,
                "average_upload_time_seconds": average_upload_time.total_seconds(),
                "average_upload_time_minutes": average_upload_time
            }
            self.image_data.append({**data})
        self.image_data = sorted(self.image_data, key=lambda x: x["epoch"])

    def save_metadata_to_csv(self):
        if not self.image_data:
            self.extract_image_metadata()
        df = pd.DataFrame(self.image_data)
        csv_output_path = self.csv_output_dir / f"{self.batch_id}_metadata.csv"
        df.to_csv(csv_output_path, index=False)

    def get_first_and_last_upload(self):
        if not self.image_data:
            self.extract_image_metadata()
        if not self.image_data:
            return None, None
        first_upload = self.image_data[0]["file_datetime_est_modified"]
        last_upload = self.image_data[-1]["file_datetime_est_modified"]
        return first_upload, last_upload

    def generate_capture_line_plot(self):
        if not self.image_data:
            self.extract_image_metadata()
        timestamps = [datetime.fromtimestamp(data["epoch"]) for data in self.image_data]
        df = pd.DataFrame({'Date': timestamps})
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], range(len(df)), marker='o', linestyle='-', color='b')

        plt.xlabel('Capture Time (EDT)')
        plt.ylabel('Image Index')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        plt.title('Capture Time Plot (based on epoch in the filename)')

        file_path = self.plot_file_base / f"capture_time_plot_{self.batch_id}.png"
        plt.savefig(file_path)
        plt.close()

    def generate_modified_line_plot(self):
        if not self.image_data:
            self.extract_image_metadata()
        timestamps = sorted([data["file_datetime_est_modified"] for data in self.image_data])

        df = pd.DataFrame({'Date': timestamps})
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], range(len(df)), linestyle='-', color='b')

        plt.xlabel('Upload Time (EDT)')
        plt.ylabel('Image Index')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()
        plt.title('Upload Time Plot (based on file modified time)')
        file_path = self.plot_file_base / f"upload_time_plot_{self.batch_id}.png"
        plt.savefig(file_path)
        plt.close()
    
    def generate_average_upload_time_plot(self):
        if not self.image_data:
            self.extract_image_metadata()
        # sort self.image_data by epoch
        self.image_data = sorted(self.image_data, key=lambda x: x["epoch"])
        timestamps = [data["average_upload_time_minutes"] for data in self.image_data]
        # convert datetime.timedeltas into seconds
        timestamps = [time.total_seconds() for time in timestamps]
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(timestamps)), timestamps, linestyle='-', color='b')

        plt.xlabel('Image Index')
        plt.ylabel('Time Difference (s)')
        plt.grid(True)
        plt.title('Capture / Upload Time Difference Plot')
        file_path = self.plot_file_base / f"upload_time_difference_plot_{self.batch_id}.png"
        plt.savefig(file_path)
        plt.close()

    def calculate_average_upload_time(self):
        if not self.image_data:
            self.extract_image_metadata()
        if len(self.image_data) <= 1:
            return 0
        timestamps = [data["epoch"] for data in self.image_data]
        time_differences = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        return sum(time_differences) / len(time_differences) if time_differences else 0

    
    def generate_pdf_report(self):
        if not self.image_data:
            self.extract_image_metadata()
        batch_id = self.batch_id
        total_images = self.calculate_total_images()
        total_size = self.calculate_total_size()
        avg_size = self.calculate_average_size()
        first_upload, last_upload = self.get_first_and_last_upload()
        partial_uploads = self.count_partial_uploads()

        pdf_output_path = str(self.output_report_dir / f"{batch_id}_report.pdf")
        c = canvas.Canvas(pdf_output_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        # Set the title of the document

        c.drawString(50, 750, f"SemiField BbotV3.1 Collection Report")
        c.drawString(50, 730, f"Batch ID: {batch_id}")
        c.drawString(50, 710, f"Total Images: {total_images}")
        c.drawString(50, 690, f"Total Size: {total_size / (1024 ** 3):.2f} GiB")
        c.drawString(50, 670, f"Average Image Size: {avg_size / 1024:.2f} KiB")
        if first_upload and last_upload:
            c.drawString(50, 650, f"First Upload: {first_upload}")
            c.drawString(50, 630, f"Last Upload: {last_upload}")
        
        if partial_uploads:
            c.drawString(50, 610, f"Partial Uploads: {partial_uploads}")

        # Add the line plot to the PDF
        capture_plot_path = self.plot_file_base / f"capture_time_plot_{batch_id}.png"
        if Path(self.plot_file_base).exists():
            c.drawImage(capture_plot_path, 5, 400, width=500//1.6, height=300//1.5)
        
        # Add the line plot to the PDF
        upload_plot_path = self.plot_file_base / f"upload_time_plot_{batch_id}.png"
        if upload_plot_path.exists():
            c.drawImage(upload_plot_path, 300, 400, width=500//1.6, height=300//1.5)

        # -----------------------------
        # Add the "Sample images" section at the bottom
        # -----------------------------
        # Set heading for sample images
        if self.local_sample_dir.exists() and list(self.local_sample_dir.glob("*.jpg")):
            c.showPage()  # Start a new page
            c.setFont("Helvetica-Bold", 14)
            page_width, page_height = letter
            left_margin = 25
            right_margin = 25
            available_width = page_width - left_margin - right_margin

            sample_heading_y = page_height - 25
            c.drawString(left_margin, sample_heading_y, "Sample images")
            
            # Define grid layout for 3 rows x 3 columns
            spacing_x = 10  # Reduced horizontal spacing
            # Calculate image width so that 3 images plus 2 gaps exactly fill the available width
            image_w = (available_width - 2 * spacing_x) / 3
            image_h = image_w  # Using a square bounding box; the image itself will preserve its aspect ratio
            spacing_y = 1  # Vertical spacing between rows
            start_x = left_margin
            start_y = sample_heading_y - 175  # Starting y position below the heading
            
            # Get up to 10 images from the batch folder (upload_directory)
            sample_images = list(self.local_sample_dir.glob("*.jpg"))
            random_sample_of_sample_imags = random.sample(sample_images, min(9, len(sample_images)))
            
            for i, image_path in enumerate(random_sample_of_sample_imags):
                row = i // 3  # 5 images per row
                col = i % 3
                x = start_x + col * (image_w + spacing_x)
                y = start_y - row * (image_h + spacing_y)
                if image_path.exists():
                    # Draw the image using the provided bounding box while preserving its aspect ratio
                    c.drawImage(str(image_path),
                                x, y,
                                width=image_w,
                                height=image_h,
                                preserveAspectRatio=True,
                                anchor='c')
        else:
            log.warning("Sample images not available")
            c.drawString(50, 350, "Sample images not available")

        # Save the PDF
        c.save()
        log.info(f"PDF report saved to {pdf_output_path}")

    def generate_report(self):
        self.extract_image_metadata()
        self.save_metadata_to_csv()

        self.generate_capture_line_plot()
        self.generate_modified_line_plot()
        self.generate_average_upload_time_plot()
        self.generate_pdf_report()


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    
    report = ImageReport(cfg)
    report.generate_report()

if __name__ == "__main__":
    main()