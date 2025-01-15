from pathlib import Path
import pandas as pd
from tqdm import tqdm

def find_files_in_batches(directory):
    """
    Recursively finds .ARW or .RAW files in batch directories and retrieves their size.
    
    Parameters:
    directory (str): Path to the root directory containing batches.
    
    Returns:
    List of dictionaries with file path, size, and batch folder name.
    """
    file_data = []

    # Convert to Path object
    directory = Path(directory)

    # Iterate through all batch directories (MD_YYYY-MM-DD format)
    directories = list(directory.rglob('*/'))
    for batch_dir in tqdm(directories):
        if batch_dir.is_dir() and batch_dir.name[:2] in ['MD', 'NC', 'TX'] and len(batch_dir.name.split('_')) == 2:
            # Look for .ARW and .RAW files
            sony_files = []
            if (batch_dir / 'SONY').exists():
                sony_files = list((batch_dir / 'SONY').glob('*'))
            files = sony_files + list(batch_dir.glob('*'))
            for file_path in tqdm(files, leave=False):
                file_size = file_path.stat().st_size
                file_data.append({
                    'File Path': str(file_path),
                    'Extension': file_path.suffix,
                    'File Size (bytes)': file_size,
                    'Batch Folder': batch_dir.name
                })
    return file_data

def save_to_csv(empty_files, output_file):
    """
    Saves the list of empty files to a CSV file.
    
    Parameters:
    empty_files (list): List of empty file paths.
    output_file (str): Path to the CSV file where results will be saved.
    """
    # Create a DataFrame
    df = pd.DataFrame(empty_files)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Empty files saved to {output_file}")

def main():
    directory = "/mnt/research-projects/s/screberg/longterm_images2/semifield-upload"
    empty_files = find_files_in_batches(directory)
    output_file = "empty_files.csv"

    if empty_files:
        save_to_csv(empty_files, output_file)
        print("Empty files found:")
    else:
        print("No empty files found.")

if __name__ == "__main__":
    main()
