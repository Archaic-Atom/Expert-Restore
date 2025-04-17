import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import DEUIRTrainDataset
from net.model import DEUIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import open_clip



class DEUIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DEUIR(dim = 32, num_blocks = [2,3,3,2], num_refinement_blocks = 2)
        self.loss_fn  = nn.L1Loss()

        checkpoint = 'daclip_ViT-B-32.pt'
        self.DA_CLIP_model, self.DA_CLIP_preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)

    def forward(self, x, y):
        return self.net(x, y)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, class_id], degrad_patch, clean_patch, DA_CLIP_degrad_patch) = batch
        with torch.no_grad():
            image_features, degra_features = self.DA_CLIP_model.encode_image(DA_CLIP_degrad_patch, control=True)
        degra_features = degra_features.float()
        restored = self.net(degrad_patch, degra_features)
        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15, max_epochs=150)

        return [optimizer],[scheduler]


def main():
    print("Options")
    print(opt)
    logger = TensorBoardLogger(save_dir = "logs/")
    model = DEUIRModel()
    
    trainset = DEUIRTrainDataset(opt, model.DA_CLIP_preprocess)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    for name, param in model.named_parameters():
         if "DA_CLIP_model" in name:
             param.requires_grad = False
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()