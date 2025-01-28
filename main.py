from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.inspect_bbotv31_unpreprocessed import main as inspect_bbotv31_unpreprocessed
from src.downscale_results import main as downscale_results
from src.utils.size_warning import main as size_warning
from src.report import main as report

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "inspect_bbotv31_unpreprocessed": inspect_bbotv31_unpreprocessed,
    "downscale_results": downscale_results,
    "report": report,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    size_warning(cfg)
    cfg = OmegaConf.create(cfg)
    # mode = cfg.mode
    tasks = cfg.tasks
    for task in tasks:
        log.info(f"Starting {task}")
        if task not in TASK_REGISTRY:
            log.error(f"Task {task} not found in task registry")
            return
        
        try:
            TASK_REGISTRY[task](cfg)
        
        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    main()