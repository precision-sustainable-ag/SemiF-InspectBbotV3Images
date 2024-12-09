import sys
from pathlib import Path
import logging

# Add the src directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf  # Do not confuse with dataclass.MISSING
from hydra.utils import get_method
import subprocess

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.pipeline)}")
    # Update the batches_in_storage.txt file
    repo_root = cfg.paths.root_dir
    subprocess.run(
        f"ls {cfg.paths.primary_storage_uploads} > {cfg.paths.batches_in_storage}",
        shell=True,
        check=True
    )
    for tsk in cfg.pipeline:
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    main()