from abc import ABC, abstractmethod
import wandb

class _BaseLogger(ABC):
    @abstractmethod
    def log(
        self,
        metric_results: dict[str, float],
        epoch_num: int,
    ) -> None: ...
    

class WandbLogger(_BaseLogger):
    def __init__(
        self,
        key: str,
        project: str,
        name: str,
        configs: dict = {},
    ):
        wandb.login(key=key)
        wandb.init(project=project, name=name, config=configs)
    
    def log(
        self,
        metric_results: dict[str, float],
        epoch_num: int,
    ) -> None:
        wandb.log(metric_results, step=epoch_num)
            
                
                