import PIL
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from data.transform import get_test_transform
from config.settings import DEVICE, FEATURE_EXTRACTOR_WEIGHTS, IMAGE_SIZE, SIMILARITY_THRESHOLD, TEST_DATA_ROOT
from torch.utils.data import DataLoader
from models.backbone import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm

from preprocess import preprocess, preprocessone

class FaceTester:
    def __init__(self, weights_path=FEATURE_EXTRACTOR_WEIGHTS, data_root=TEST_DATA_ROOT):
        self.transform = get_test_transform()
        self.data_root = data_root
        self.weights_path = weights_path
        
        self.extractor = FeatureExtractor(pretrained=False)
        if os.path.exists(weights_path):
            weights = torch.load(weights_path, map_location='cpu')
            self.extractor.load_state_dict(weights)
            print(f"成功加载权重: {weights_path}")
        else:
            print(f"警告: 权重文件不存在 {weights_path}")
        
        self.extractor.eval()
        self.extractor = self.extractor.to(DEVICE)
        
        # 构建人员数据库
        self.people_database = {}
        self.label_to_name = {}
        self.name_to_label = {}
        
    def build_database(self):
        print("正在构建人员特征数据库...")
        
        # 使用ImageFolder加载数据
        dataset = ImageFolder(self.data_root, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        
        # 标签到名称的映射
        self.label_to_name = {v: k for k, v in dataset.class_to_idx.items()}
        self.name_to_label = dataset.class_to_idx
        
        all_features = {}
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="提取特征"):
                images = images.to(DEVICE)
                features = self.extractor(images)
                features = features.cpu().numpy()
                
                for i, label in enumerate(labels):
                    label = label.item()
                    if label not in all_features:
                        all_features[label] = []
                    all_features[label].append(features[i])
        
        # 平均特征向量
        for label, features in all_features.items():
            features_array = np.array(features)
            mean_feature = np.mean(features_array, axis=0)
            
            # 归一
            mean_feature = mean_feature / np.linalg.norm(mean_feature)
            
            self.people_database[label] = {
                'mean_feature': mean_feature,
                'sample_count': len(features),
                'name': self.label_to_name[label]
            }
        
        print(f"数据库构建完成，共 {len(self.people_database)} 个人")
    
    def recognize(self, img_path, top_k=5):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        img = PIL.Image.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0).to(DEVICE)
        
        # 提取特征
        with torch.no_grad():
            feature = self.extractor(img)
            feature = feature.cpu().numpy()[0]
        
        # 归一化
        feature = feature / np.linalg.norm(feature)
        
        # 计算与数据库中所有特征的相似度
        similarities = {}
        for label, data in self.people_database.items():
            similarity = cosine_similarity([feature], [data['mean_feature']])[0][0]
            similarities[label] = {
                'similarity': similarity,
                'name': data['name']
            }
        
        sorted_similarities = sorted(
            similarities.items(), 
            key=lambda x: x[1]['similarity'], 
            reverse=True
        )
        
        best_match_label, best_match_data = sorted_similarities[0]
        best_similarity = best_match_data['similarity']
        best_match_name = best_match_data['name']
        
        top_results = []
        for i, (label, data) in enumerate(sorted_similarities[:top_k]):
            top_results.append({
                'rank': i + 1,
                'label': label,
                'name': data['name'],
                'similarity': data['similarity']
            })
        
        if best_similarity >= SIMILARITY_THRESHOLD:
            return best_match_label, best_similarity, top_results, best_match_name
        else:
            return None, best_similarity, top_results, "Unknown"

    def save_database(self, path):
        save_data = {
            'people_database': self.people_database,
            'label_to_name': self.label_to_name,
            'name_to_label': self.name_to_label,
            'metadata': {
                'num_people': len(self.people_database),
                'total_samples': sum([v['sample_count'] for v in self.people_database.values()])
            }
        }
        
        torch.save(save_data, path)
        print(f"人员数据库已保存到: {path}")
    
    def load_database(self, path):
        if not os.path.exists(path):
            print(f"数据库文件不存在: {path}")
            return False
        
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        self.people_database = data['people_database']
        self.label_to_name = data['label_to_name']
        self.name_to_label = data['name_to_label']
        
        print(f"人员数据库已加载: {len(self.people_database)} 个人")
        return True

if __name__ == "__main__":
    preprocess(input_dir="testdataraw", output_dir="testdata", output_size=(IMAGE_SIZE, IMAGE_SIZE))
    tester = FaceTester()
    tester.build_database()
    import argparse
    parser = argparse.ArgumentParser(description="人脸识别测试")
    parser.add_argument(type=str, default='test.jpg', help='测试图像路径', dest='img')
    args = parser.parse_args()
    test_image_path = args.img
    if os.path.exists(test_image_path):
        print(f"正在识别图像: {test_image_path}")
        preprocessone(test_image_path,'aligned_test.jpg',output_size=(IMAGE_SIZE, IMAGE_SIZE))
        print('预处理图像到 aligned_test.jpg')
        label, similarity, top_matches, name = tester.recognize('aligned_test.jpg')
        print(f"识别结果: {name} (相似度: {similarity:.4f})")
        print("前5个匹配结果:")
        for match in top_matches:
            print(f"  {match['rank']}. {match['name']}: {match['similarity']:.4f}")
    else:
        print(f"测试图像不存在: {test_image_path}")