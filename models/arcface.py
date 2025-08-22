import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .backbone import FeatureExtractor
from config.settings import ARCFACE_S, ARCFACE_M, EASY_MARGIN

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=ARCFACE_S, m=ARCFACE_M, easy_margin=EASY_MARGIN):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, pretrained=True):
        super(ArcFaceModel, self).__init__()
        
        self.feature_extractor = FeatureExtractor(embedding_size, pretrained)
        self.classifier = ArcMarginProduct(embedding_size, num_classes)
    
    def forward(self, x, labels=None):
        embedding = self.feature_extractor(x)
        
        if labels is not None:
            output = self.classifier(embedding, labels)
            return output, embedding
        else:
            return embedding