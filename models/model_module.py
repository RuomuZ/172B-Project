import sys
import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics
import numpy as np
sys.path.append(".")
from models.resnet_transfer import FCNResnetTransfer

class MGZSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        super(MGZSegmentation, self).__init__()
        self.save_hyperparameters("model_type", "in_channels", "out_channels", 
                 "learning_rate", "model_params")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        self.model_params = model_params
        if model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels,out_channels, kwargs=model_params)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=4)
        self.loss_function = nn.CrossEntropyLoss()
            
    def forward(self, X):
        X = torch.nan_to_num(X)
        X = self.model(X)
        return X
    
    def training_step(self, batch, batch_idx):
        sat_img, mask = batch
        logits = self(sat_img)
        print(sat_img)
        print(mask)
        result = self.loss_function(logits, mask.long())
        self.log("loss", result)
        print(result)
        return result
    
    
    def validation_step(self, batch, batch_idx):
        sat_img, mask = batch
        pred = self(sat_img)
        #acc = self.accuracy(pred, mask)
        loss = self.loss_function(pred.float(), mask)
        #self.log_dict({"acc": acc, "val_loss": loss}, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optm = Adam(self.model.parameters(), lr=self.learning_rate)
        return optm
