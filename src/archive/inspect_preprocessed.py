import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, cfg):
        self.primary_storage = Path(cfg.paths.primary_storage)
        self.upload_batches = sorted(list(Path(self.primary_storage, "semifield-upload").glob("*")))
        self.developed_batches = sorted(list(Path(self.primary_storage, "semifield-developed-images").glob("*")))
        self.preprocessed_batches = set([b.name for b in self.upload_batches]) & set([b.name for b in self.developed_batches])
        self.batch_info = []

    def process_batches(self):
        for batch in self.preprocessed_batches:
            upload_batch_path = Path(self.primary_storage, "semifield-upload", batch)
            
            if (upload_batch_path / "SONY").exists():
                upload_batch_path = upload_batch_path / "SONY"
            
            developed_batch_path = Path(self.primary_storage, "semifield-developed-images", batch)
            
            upload_date = datetime.fromtimestamp(upload_batch_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            developed_date = datetime.fromtimestamp(Path(developed_batch_path, "images").stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            num_arw_files = len(list(upload_batch_path.glob("*.ARW")))
            num_jpg_files = len(list(Path(developed_batch_path, "images").glob("*.jpg")))
            
            plant_detections_exists = (Path(developed_batch_path, "plant-detections").exists())
            
            self.batch_info.append({
                "batch": batch,
                "num_arw_files": num_arw_files,
                "num_jpg_files": num_jpg_files,
                "jpg_arw_match": abs(num_arw_files - num_jpg_files) <= 3,
                "plant_detections_folder_exists": plant_detections_exists,
                "upload_lastmodified_date": upload_date,
                "developed_last_modified_date": developed_date,
                "upload_path": str(upload_batch_path),
                "developed_path": str(developed_batch_path),
            })

    def output_to_csv(self, csv_file_path):
        df = pd.DataFrame(self.batch_info).sort_values(by="developed_last_modified_date", ascending=False)
        df.to_csv(csv_file_path, index=False)


    def create_bar_graph(self, plot_path):
        df = pd.DataFrame(self.batch_info)
        df['developed_last_modified_date'] = pd.to_datetime(df['developed_last_modified_date'])
        df = df[df['jpg_arw_match']]

        # Group by week
        df['week'] = df['developed_last_modified_date'].dt.to_period('W').apply(lambda r: r.start_time)
        df['week'] = df['week'].dt.strftime('%Y-%m-%d')
        weekly_counts = df.groupby('week').size()

        # Plotting
        plt.figure(figsize=(10, 6))
        weekly_counts.plot(kind='bar')
        
        plt.xlabel('Week')
        plt.ylabel('Number of Batches')
        plt.title('Number of SemiField Batches Preprocessed by Week')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path)

def main(cfg):
    processor = BatchProcessor(cfg)
    processor.process_batches()
    reports = cfg.paths.reports
    date_str = datetime.now().strftime('%Y%m%d')
    date_reports_dir = Path(reports, date_str, "preprocessed")
    date_reports_dir.mkdir(parents=True, exist_ok=True)
    csv_file_path = Path(date_reports_dir, f'preprocessed_batch_info.csv')
    processor.output_to_csv(csv_file_path)
    plot_path = Path(date_reports_dir, f'preprocessed_batch_info.png')
    processor.create_bar_graph(plot_path)