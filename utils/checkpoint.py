import torch
import os
from datetime import datetime
from config.settings import CHECKPOINT_DIR, TIMESTAMP

class CheckpointManager:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, loss, accuracy, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}_{TIMESTAMP}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        #最佳模型额外保存
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, 
                f'best_model_{TIMESTAMP}.pth'
            )
            torch.save(checkpoint, best_path)
            
        return checkpoint_path
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
    
    def get_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            return None
            
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])