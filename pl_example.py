import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 定义LightningModule
class FinetuneRepresentationModel(pl.LightningModule):
    def __init__(self, embedding_model, representation_model, num_classes=10, learning_rate=1e-3):
        super(FinetuneRepresentationModel, self).__init__()
        self.save_hyperparameters()

        # 加载预训练的 embedding model 和 representation model
        self.embedding_model = embedding_model
        self.representation_model = representation_model

        # 定义 output head，通常是一个全连接层
        self.output_model = nn.Linear(representation_model.output_dim, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # 输入先通过 embedding model，然后 representation model，最后是 output head
        embedding = self.embedding_model(x)
        representation = self.representation_model(embedding)
        output = self.output_model(representation)
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# 定义数据模块
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # 定义数据增强和预处理
        self.transform = transforms.Compose([
            transforms.Resize(224),  # 适应模型的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用ImageNet的均值和标准差
                                 std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = datasets.CIFAR10(root=self.data_dir, train=True,
                                              transform=self.transform, download=True)
        self.val_dataset = datasets.CIFAR10(root=self.data_dir, train=False,
                                            transform=self.transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4)

# 主训练函数
def main():
    # 加载预训练模型的 embedding model 和 representation model
    # 假设你有预训练的这两个部分，使用你的方式载入
    embedding_model = torch.load('path_to_pretrained_embedding_model.pth')
    representation_model = torch.load('path_to_pretrained_representation_model.pth')

    # 初始化数据模块
    data_module = CIFAR10DataModule()

    # 初始化模型，添加新的 output head
    model = FinetuneRepresentationModel(embedding_model=embedding_model,
                                        representation_model=representation_model,
                                        num_classes=10, learning_rate=1e-3)

    # 定义回调：保存最好的模型和早停
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='representation_model-cifar10-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    early_stop_callback = EarlyStopping(monitor='val_acc', patience=5, mode='max')

    # 初始化Trainer
    trainer = pl.Trainer(
        max_epochs=20,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        progress_bar_refresh_rate=20,
    )

    # 开始训练
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()
