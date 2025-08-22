import torch.optim as optim
from torch.optim import lr_scheduler
from config.settings import LR_SCHEDULER, STEP_SIZE, GAMMA, LR_PATIENCE, LR_FACTOR, LR_MIN
import torch
def get_lr_scheduler(optimizer, scheduler_type=LR_SCHEDULER):
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(
            optimizer, 
            step_size=STEP_SIZE, 
            gamma=GAMMA
        )
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=STEP_SIZE, 
            eta_min=LR_MIN
        )
    elif scheduler_type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=LR_PATIENCE, 
            factor=LR_FACTOR, 
            min_lr=LR_MIN
        )
    else:
        raise ValueError(f"未知的学习率调度器类型: {scheduler_type}")