import argparse
import os
from torch.utils.data import DataLoader, random_split
from config.settings import DATA_ROOT, BATCH_SIZE, NUM_EPOCHS
from data.dataset import PreloadedDataset
from data.transform import get_train_transform
from models.arcface import ArcFaceModel
from training.trainer import Trainer
from utils.logger import logger
def main():
    parser = argparse.ArgumentParser(description='训练人脸识别模型')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='批次大小')
    parser.add_argument('--validation-split', type=float, default=0.25, help='验证集比例')
    parser.add_argument('--h5-path', type=str, default='dataset.h5', help='HDF5文件路径（如果使用预加载数据）')
    args = parser.parse_args()
    
    logger.info("开始准备训练数据")
    
    transform = get_train_transform()
    if args.h5_path and os.path.exists(args.h5_path):
        dataset = PreloadedDataset(h5_path=args.h5_path, transform=transform)
    else:
        dataset = PreloadedDataset(
            DATA_ROOT, 
            transform=transform
        )
        dataset.to_h5('./dataset.h5', compression='gzip', compression_opts=9)
    
    val_size = int(len(dataset) * args.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_classes = dataset.get_num_classes()
    model = ArcFaceModel(num_classes=num_classes)
    
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"类别数量: {num_classes}")
    
    trainer = Trainer(model, train_loader, val_loader, num_classes)
    trainer.train(num_epochs=args.epochs, resume_checkpoint=args.resume)
    
    logger.info("训练完成")

if __name__ == "__main__":
    main()