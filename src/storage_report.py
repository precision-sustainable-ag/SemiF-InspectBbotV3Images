from pathlib import Path
from omegaconf import DictConfig
import logging
import os
from datetime import datetime
from pathlib import Path
from utils.utils import get_batch_info

log = logging.getLogger(__name__)

def ensure_directory(path):
    """Ensure that the directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def calculate_folder_size(folder):
    """Calculate the total size of a folder in bytes."""
    total_size = 0
    for path, _, files in os.walk(folder):
        for file in files:
            total_size += Path(path, file).stat().st_size
    return total_size

def generate_storage_report(batch_name, batch_par):
    """Generate storage statistics for a batch."""
    batch_dir = Path(batch_par, batch_name)

    # Get image statistics
    jpgs, raws = get_batch_info(batch_dir, print_info=False)
    jpg_count, raw_count = len(jpgs), len(raws)
    jpg_size = sum([x.stat().st_size for x in jpgs]) / (1024**2)  # MiB
    raw_size = sum([x.stat().st_size for x in raws]) / (1024**2)  # MiB

    # Calculate total folder size
    
    total_size_bytes = calculate_folder_size(batch_dir)
    total_size_gib = total_size_bytes / (1024**3)  # GiB
 
    report_entry = (
        f"{batch_name} ({total_size_gib:.2f} GiB) | {jpg_count} jpgs ({jpg_size:.2f} MiB) | {raw_count} arws ({raw_size:.2f} MiB)\n"

    )
    return report_entry, total_size_bytes

def process_batches(batch_file, batch_par, report_file):
    """Process multiple batches and generate a combined report."""
    # Ensure the reports directory exists
    ensure_directory(report_file.parent)

    # Read batch names from the input file
    with open(batch_file, "r") as f:
        batches = [line.strip() for line in f if line.strip()]

    # Generate reports for all batches
    report_content = "Storage Report for All Batches\n"
    report_content += "=" * 40 + "\n"
    
    total_storage_size_bytes = 0
    
    for batch_name in batches:
        print(f"Processing batch: {batch_name}")
        try:
            entry, size_bytes = generate_storage_report(batch_name, batch_par)
            report_content += entry
            total_storage_size_bytes += size_bytes  # Accumulate size in bytes
            
        except Exception as e:
            report_content += f"Error processing batch {batch_name}: {str(e)}\n"
            report_content += "-" * 40 + "\n"
    # Convert total size to TiB and append to report
    total_size_tib = total_storage_size_bytes / (1024**4)  # TiB
    report_content += f"\nTotal Storage Size of Directory: {total_size_tib:.2f} TiB\n"

    # Save the combined report
    with report_file.open("w") as f:
        f.write(report_content)

    print(f"Combined storage report saved to: {report_file}")

def main(cfg: DictConfig) -> None:
    batch_file = cfg.paths.batches_in_storage
    batch_par = cfg.paths.primary_storage
    
    timestamp = datetime.now().strftime("%Y%m%d")
    report_file = Path(cfg.paths.reports, timestamp, "storage_report.txt")
    # Generate the combined report
    process_batches(Path(batch_file), Path(batch_par), Path(report_file))