import torch
import torch.nn.functional as F

def kernel_contrast_loss(kernels_vec, task_ids, alpha=0.01):
    total_loss = 0.0
    for idx in range(len(kernels_vec)):
        kernels = kernels_vec[idx]
        total_loss += alpha * contrast_loss(kernels, task_ids)
    return total_loss

def contrast_loss(kernels, task_ids):
    B, C, K, _ = kernels.shape
    device = kernels.device
    
    kernels_flat = kernels.view(B, -1)
    kernels_norm = F.normalize(kernels_flat, dim=1)

    intra_loss = 0.0
    inter_loss = 0.0
    intra_count = 0
    inter_count = 0

    for i in range(B):
        for j in range(i + 1, B):
            sim = torch.sum(kernels_norm[i] * kernels_norm[j])  # Cosine similarity
            if task_ids[i] == task_ids[j]:
                intra_loss += (1 - sim)
                intra_count += 1
            else:
                inter_loss += sim
                inter_count += 1

    if intra_count > 0:
        intra_loss = intra_loss / intra_count
    if inter_count > 0:
        inter_loss = inter_loss / inter_count

    total_loss = intra_loss + inter_loss
    return total_loss