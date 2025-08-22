import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size=512, pretrained=True):
        super(FeatureExtractor, self).__init__()
        
        # 使用预训练ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 添加自定义嵌入层
        self.embedding = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        embedding = self.embedding(x)
        embedding = self.bn(embedding)
        embedding = self.dropout(embedding)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding