import logging
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from signal import SIGINT

log = logging.getLogger(__name__)

def format_size(bytes_size: float) -> str:
    """
    Format the size in bytes to the most appropriate unit: GiB, MiB, or TiB.

    Args:
        bytes_size (float): The size in bytes.

    Returns:
        str: The formatted size string with the appropriate unit.
    """
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024

    return f"{bytes_size:.2f} PiB"

def get_directory_size(directory: Path) -> float:
    """
    Calculate the total size of a directory in bytes using pathlib.

    Args:
        directory (Path): The path to the directory.

    Returns:
        float: The size of the directory in bytes.
    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")
    
    return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())

def check_directory_size(directory: Path, size_limit_gib: float) -> None:
    """
    Check if the directory size exceeds the given limit.

    Args:
        directory (Path): The path to the directory.
        size_limit_gib (float): The maximum allowed size in GiB.

    Raises:
        SystemExit: If the directory size exceeds the limit.
    """
    current_size_bytes = get_directory_size(directory)
    current_size_gib = current_size_bytes / (1024 ** 3)  # Convert bytes to GiB
    formatted_size = format_size(current_size_bytes)

    if current_size_gib > size_limit_gib:
        logging.warning(f"Directory {directory} exceeds the size limit of {size_limit_gib:.2f} GiB. Current size: {formatted_size}.")
        sys.exit(f"Script stopped: Directory size is too large ({formatted_size}).")

    logging.info(f"Directory size: {formatted_size}. Size is within the limit of {size_limit_gib:.2f} GiB.")

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    directory_to_check = Path(cfg.paths.local_upload)
    # Specify the directory to check and the size limit (in MiB)
    size_limit = 100.0  # Example limit in GiB (adjust as needed)

    try:
        check_directory_size(directory_to_check, size_limit)
        logging.info("Directory size is within the acceptable limit. Proceeding with the script...")
        # Add the rest of your script logic here
    except ValueError as e:
        logging.error(e)
        sys.exit("Script stopped due to invalid directory.")
 

if __name__ == "__main__":
    main()
