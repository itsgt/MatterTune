# %%
from __future__ import annotations

import os
from pathlib import Path

from mattertune.finetune.main import MatterTuner, MatterTunerConfig

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    config = MatterTunerConfig.model_validate_json(
        Path("/workspaces/MatterTune/config.json").read_text()
    )

    MatterTuner(config).tune()


if __name__ == "__main__":
    main()

# %%
