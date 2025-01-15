from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.inspect_bbotv31_unpreprocessed import main as inspect_bbotv31_unpreprocessed

log = logging.getLogger(__name__)

# Define a registry of tasks
TASK_REGISTRY = {
    "inspect_bbotv31_unpreprocessed": inspect_bbotv31_unpreprocessed,
    # Add more tasks here as needed
}

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    mode = cfg.mode
    log.info(f"Starting {mode}")
    
    if mode not in TASK_REGISTRY:
        log.error(f"Task {mode} not found in task registry")
        return
    
    try:
        TASK_REGISTRY[mode](cfg)
    
    except Exception as e:
        log.exception("Failed")
        return


if __name__ == "__main__":
    main()