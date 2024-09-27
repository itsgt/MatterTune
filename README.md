

data_module = ABC-MatterTuneBaseDataModule

backbone = load_backbone()
backbone = GOCStyleBB(backbone)

output_heads = list[OutputHeadConfig]

finetune = FinetuneModuleBase(
    backbone
    output_heads
)

trainer = pl.Trainer(finetune)

<!-- trainer : pl.Lignthning Module : FinetuneModuleBase

FinetuneModuleBase
- __init__
- forward
- train_step
- val_step
- test_step
- predict_step
- save
- load
... 
- self.lr_scheduler: basic implementation + protocol, StepLR|Platauo|Cosine
- self.optimizer: basic implementation
- self.monitor: basic implementation

trainer.fit(data_module)

trainer -> Calculator

class Calculator
    def calculate(atoms)
        atoms -> datamodule
        trainer.predict(datamodule)