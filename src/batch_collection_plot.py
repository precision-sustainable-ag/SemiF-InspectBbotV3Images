import re
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import DictConfig
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm

def parse_batch_name(batch_name):
    """Extract site and date from the batch name."""
    pattern = r'^([A-Z]{2})_(\d{4}-\d{2}-\d{2})$'
    match = re.match(pattern, batch_name)
    if match:
        site, date_str = match.groups()
        date = datetime.strptime(date_str, "%Y-%m-%d")
        return site, date
    else:
        raise ValueError(f"Invalid batch name format: {batch_name}")

def read_batches_from_file(batch_file):
    """Read batch names from the input file."""
    batches = []
    with open(batch_file, "r") as f:
        for line in tqdm(f):
            batch_name = line.strip()
            if batch_name:
                try:
                    site, date = parse_batch_name(batch_name)
                    batches.append((site, date, batch_name))
                except ValueError as e:
                    print(e)
    return batches

def get_image_counts(batch_par, batch_name):
    """Get the number of JPG and RAW images in a batch."""
    batch_dir = Path(batch_par, batch_name)
    jpgs = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.JPG"))
    raws = list(batch_dir.glob("*.arw")) + list(batch_dir.glob("*.ARW")) + list(batch_dir.glob("*.RAW"))
    return len(jpgs), len(raws)

def plot_batch_collection(batches, batch_par, output_dir):
    """Generate grouped bar graphs for batch collections."""
    # Organize data by site
    site_data = defaultdict(list)
    for site, date, batch_name in batches:
        jpg_count, raw_count = get_image_counts(batch_par, batch_name)
        site_data[site].append((date, jpg_count, raw_count))

    # Create a bar graph for each site
    for site, data in site_data.items():
        # Sort data by date
        data.sort(key=lambda x: x[0])
        dates = [x[0] for x in data]
        jpg_counts = [x[1] for x in data]
        raw_counts = [x[2] for x in data]

        # Set bar width and positions for grouped bars
        bar_width = 0.35
        x = np.arange(len(dates))  # X-axis positions for the groups

        # Create the plot
        plt.figure(figsize=(14, 7))
        plt.bar(x - bar_width / 2, jpg_counts, width=bar_width, color='cornflowerblue', edgecolor='black', label='JPG Images')
        plt.bar(x + bar_width / 2, raw_counts, width=bar_width, color='lightcoral', edgecolor='black', label='RAW Images')

        # X-axis formatting for dates
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Tick every 5 days
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(x, [date.strftime('%Y-%m-%d') for date in dates], rotation=85, ha='center')

        plt.xlabel('Collection Date')
        plt.ylabel('Number of Images')
        plt.title(f'Batch Collection Timeline for {site}')
        plt.legend()

        # Save the plot
        output_path = Path(output_dir, f"{site}_batch_collection_bar.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path)
        # plt.show()
        plt.close()

        print(f"Saved plot: {output_path}")

def main(cfg: DictConfig) -> None:
    """Main function to generate batch collection plots."""
    batch_file = cfg.paths.batches_in_storage
    batch_par = cfg.paths.primary_storage_uploads

    timestamp = datetime.now().strftime("%Y%m%d")
    output_dir = Path(cfg.paths.reports, timestamp, "plots")
    
    batches = read_batches_from_file(batch_file)
    plot_batch_collection(batches, batch_par, output_dir)
