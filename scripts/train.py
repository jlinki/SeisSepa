import _bootstrap  # noqa: F401
import hydra
from hydra.core.hydra_config import HydraConfig

from seissepa.train import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config):
    run(config, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
