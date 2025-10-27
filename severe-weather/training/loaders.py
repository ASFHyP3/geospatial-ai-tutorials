from collections.abc import Callable

import torch


class MultimodalNormalize(Callable):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, batch):
        for m in self.means.keys():
            if m not in batch['image']:
                continue
            image = batch['image'][m]
            if len(image.shape) == 5:
                # B, C, T, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1, 1)
            elif len(image.shape) == 4:
                # B, C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1)
            elif len(image.shape) == 2 or len(image.shape) == 1:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)
            else:
                msg = (
                        f'Expected batch with 4 dimensions (B, C, H, W), sample with 2 dimensions (H, W) '
                        f'or a single channel, but got {image.shape}'
                        )
                raise Exception(msg)

            batch['image'][m] = (image - means) / stds
        return batch
