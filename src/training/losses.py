"""
This code is adapted from https://github.com/Mathux/TMR
"""

import torch
import torch.nn.functional as F


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_token=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        # Filtering too close values
        if sent_token is not None and self.threshold_selfsim:
            # Put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Compute caption self-similarity
            sent_token = sent_token / sent_token.norm(dim=1, keepdim=True)
            selfsim = sent_token @ sent_token.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            # Mask them by putting -inf in the sim_matrix
            sim_matrix[selfsim_nodiag > real_threshold_selfsim] = -torch.inf

        labels = torch.arange(bs, device=device)
        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
