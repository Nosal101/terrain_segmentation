import torch
import lightning.pytorch as pl
import torchmetrics
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from neptune.types import File

class DefaultSegmentationModel(pl.LightningModule):

    def __init__(self, num_classes, input_channels=3, learning_rate=1e-3, encoder_name="resnet34", T_MAX=100):
        super().__init__()
        self.T_MAX = T_MAX
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.default_device = torch.device("cpu")

        if torch.cuda.is_available():
            self.default_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.default_device = torch.device("mps")
        
        # Ładujemy pretrenowany model YOLOv5 z detekcją
        self.network = smp.Unet(
                encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_function = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.save_hyperparameters()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, image):
        # image = image.to(self.default_device)
        image = (image - self.mean) / self.std
        return self.network(image)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch)

        self.training_step_outputs.append(train_loss_info)

        return train_loss_info
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch)

        self.validation_step_outputs.append(valid_loss_info)

        return valid_loss_info
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return
    
    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, 'test')
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def shared_step(self, batch, stage="train"):
        inputs, labels = batch
        outputs = self(inputs)
        outputs = outputs.squeeze(1)
        loss = self.loss_function(outputs, labels.float())
        self.log("metrics/batch/loss", loss, prog_bar=True)

        prob_mask =  outputs.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        if stage == "test":
            amount = pred_mask.shape[0]
            for i in range(amount):
                image = inputs.cpu()[i, :, :, :].squeeze(0)
                image = image.permute(1, 2, 0)
                self.logger.experiment[f"test/prediction"].append(File.as_image(image))
                self.logger.experiment[f"test/prediction"].append(File.as_image(pred_mask.cpu()[i, :, :]))
                self.logger.experiment[f"test/prediction"].append(File.as_image(labels.cpu()[i, :, :]))

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), labels.long(), mode="binary"
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        self.log(f"metrics/epoch/{stage}/per_image_iou", per_image_iou, prog_bar=True)

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.log(f"metrics/epoch/{stage}/dataset_iou", dataset_iou, prog_bar=True)
        # metrics = {
        #     f"{stage}_per_image_iou": per_image_iou,
        #     f"{stage}_dataset_iou": dataset_iou,
        # }

        # self.log_dict(metrics, prog_bar=True)