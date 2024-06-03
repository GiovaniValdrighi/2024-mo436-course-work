from torch import nn
import torch


class TFC(nn.Module):
    def __init__(self, 
            encoder_t,
            encoder_f,
            projector_t,
            projector_f,):
        super(TFC, self).__init__()

        self.encoder_t = encoder_t
        self.encoder_f = encoder_f
        self.projector_t = projector_t
        self.projector_f = projector_f

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning"""


class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
