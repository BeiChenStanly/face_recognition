import torch
import torch.optim as optim
from tqdm import tqdm
import time
import os
from config.settings import DEVICE, EARLY_STOPPING_PATIENCE, FEATURE_EXTRACTOR_WEIGHTS, NUM_EPOCHS, LEARNING_RATE, TIMESTAMP, WEIGHT_DECAY, CHECKPOINT_FREQUENCY
from utils.logger import logger, progress_logger
from utils.checkpoint import CheckpointManager
from training.lr_scheduler import get_lr_scheduler
from utils.visualization import plot_training_curves

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, num_classes=None):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        self.scheduler = get_lr_scheduler(self.optimizer)
        self.checkpoint_manager = CheckpointManager()
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        logger.info("训练器初始化完成")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels, _) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            outputs, embeddings = self.model(images, labels)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 记录进度 Slurm作业检查
            progress_logger.info(
                f"Epoch: {epoch}, Batch: {batch_idx}/{len(self.train_loader)}, "
                f"Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        epoch_time = time.time() - start_time
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        logger.info(
            f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, "
            f"Train Acc: {epoch_accuracy:.2f}%, Time: {epoch_time:.2f}s, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, epoch):
        if not self.val_loader:
            return float('inf'), 0.0
            
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs, embeddings = self.model(images, labels)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        return val_loss, val_accuracy, is_best
    
    def train(self, num_epochs=NUM_EPOCHS, resume_checkpoint=None):
        start_epoch = 0
        
        if resume_checkpoint:
            try:
                start_epoch, _, _ = self.checkpoint_manager.load_checkpoint(
                    self.model, self.optimizer, self.scheduler, resume_checkpoint
                )
                logger.info(f"从检查点恢复训练，起始epoch: {start_epoch}")
            except Exception as e:
                logger.error(f"加载检查点失败: {e}")
        
        for epoch in range(start_epoch + 1, num_epochs + 1):
            train_loss, train_accuracy = self.train_epoch(epoch)
            
            val_loss, val_accuracy, is_best = self.validate(epoch)
            
            # 更新学习率
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            if epoch % CHECKPOINT_FREQUENCY == 0 or is_best:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, val_loss, val_accuracy, is_best
                )
                logger.info(f"检查点已保存: {checkpoint_path}")
            
            plot_training_curves(
                self.train_losses, self.val_losses, 
                self.train_accuracies, self.val_accuracies,
                self.learning_rates
            )
            
            # 早停
            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logger.info(f"早停触发，在epoch {epoch}停止训练")
                break
        
        final_model_path = os.path.join('models', f'final_model_{TIMESTAMP}.pth')
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")

        # 保存特征提取器
        torch.save(self.model.feature_extractor.state_dict(), FEATURE_EXTRACTOR_WEIGHTS)
        
        plot_training_curves(
            self.train_losses, self.val_losses, 
            self.train_accuracies, self.val_accuracies,
            self.learning_rates
        )
        
        return self.model