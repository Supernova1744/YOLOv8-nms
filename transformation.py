import torch
from torch import nn

class Transform(nn.Module):
    def forward(self, idxTensor, boxes, scores):
        batches = idxTensor[:, 0]
        bbox_result = self.gather(boxes, idxTensor)

        score_intermediate_result = self.gather(scores, idxTensor).max(axis=-1)
        score_result = score_intermediate_result.values
        classes_result = score_intermediate_result.indices

        concatenated = torch.concat([
                                    bbox_result[0],
                                    score_result.T,
                                    classes_result.T], -1)

        return concatenated, batches

    def gather(self, target, idxTensor):
        pick_indices = idxTensor[..., -1:].repeat(1, target.shape[1])
        if len(pick_indices.shape) == 2:
            pick_indices = pick_indices.unsqueeze(0)
        return torch.gather(target.permute(0, 2, 1), 1, pick_indices)
