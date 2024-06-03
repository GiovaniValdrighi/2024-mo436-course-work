import sys

sys.path.append("../../")

from torch import nn
import torch
import lightning as L

import models.tfc as tfc
import models.autoencoder as ae
import models.mlp as mlp


### -------------------------------------------------------------------------------


# This function must save the weights of the pretrained model
def pretext_save_backbone_weights(pretext_model, checkpoint_filename):
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.backbone.state_dict(), checkpoint_filename)


# This function must instantiate and configure the datamodule for the pretext task
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning DataModule.
def build_pretext_datamodule() -> L.LightningDataModule:
    # Build the transform object
    transform = BarlowTwinsTransforms()
    # Create the datamodule
    return F3SeismicDataModule(
        root_dir="../../data/", batch_size=8, transform=transform
    )


# This function must instantiate and configure the pretext model
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure it returns a Lightning model.
def build_pretext_model() -> L.LightningModule:
    # Build the backbone
    backbone = ae.AutoencoderMLP([360, 256, 128, 64])
    # Build the projection head
    projection_head = mlp.MultiLayerPerceptron(input_features=64, hidden_size=64, num_classes=6)
    # Build the loss function for the pretext
    loss_fn = ...
    # Build the pretext model
    return ...


# This function must instantiate and configure the lightning trainer
# with the best parameters found for the seismic/HAR task.
# You might change this code, but must ensure you return a Lightning trainer.
def build_lightning_trainer() -> L.Trainer:
    return L.Trainer(
        accelerator="gpu",
        max_epochs=1,
        max_steps=30,
        enable_checkpointing=False,
        logger=False,
    )


# This function must not be changed.
def main(SSL_technique_prefix):

    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_model = build_pretext_model()
    pretext_datamodule = build_pretext_datamodule()
    lightning_trainer = build_lightning_trainer()

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)

    # Save the backbone weights
    output_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    pretext_save_backbone_weights(pretext_model, output_filename)


if __name__ == "__main__":
    SSL_technique_prefix = "TFC"
    main(SSL_technique_prefix)
