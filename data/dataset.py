import os
from re import sub
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import h5py
import json
from config.settings import DATA_ROOT, IMAGE_PER_SUBJECT, PRELOAD_TO_GPU, PRELOAD_TO_CPU, PRELOAD_BATCH_SIZE
from data.transform import get_train_transform

class PreloadedDataset(Dataset):
    def __init__(self, root_dir=None, transform=None, h5_path=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_dict = {}
        self.data_tensor = None
        self.label_tensor = None
        self.h5_path = h5_path
        
        # 如果提供了HDF5路径，从HDF5加载
        if h5_path is not None and os.path.exists(h5_path):
            self._load_from_h5()
        elif root_dir is not None:
            # 否则从原始数据加载
            self._prepare_file_list()
            self._preload_data()
        else:
            raise "必须提供 root_dir 或 h5_path"
    
    def _prepare_file_list(self):
        subjects = [d for d in os.listdir(self.root_dir) 
                   if os.path.isdir(os.path.join(self.root_dir, d))]
        label_idx = 0
        
        for subject in subjects:
            subject_path = os.path.join(self.root_dir, subject)
            print(subject_path)
            for root, _, files in os.walk(subject_path):
                i = 0
                for i in range(len(files)):
                    file = files[i]
                    if file.endswith('.jpg'):
                        img_path = os.path.join(root, file)
                        self.samples.append(img_path)
                        
                        if subject not in self.label_dict:
                            self.label_dict[subject] = label_idx
                            label_idx += 1
                        self.labels.append(self.label_dict[subject])
                    if i >= IMAGE_PER_SUBJECT: 
                        break
    
    def _preload_data(self):
        device = torch.device('cuda' if PRELOAD_TO_GPU else 'cpu')
        print(f"预加载 {len(self.samples)} 张图像到 {device}...")
        
        # 分批处理以避免内存溢出
        num_batches = (len(self.samples) + PRELOAD_BATCH_SIZE - 1) // PRELOAD_BATCH_SIZE
        batch_tensors = []
        
        for batch_idx in tqdm(range(num_batches), desc="预加载批次"):
            start_idx = batch_idx * PRELOAD_BATCH_SIZE
            end_idx = min((batch_idx + 1) * PRELOAD_BATCH_SIZE, len(self.samples))
            
            batch_images = []
            for i in range(start_idx, end_idx):
                img_path = self.samples[i]
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                batch_images.append(image)
            
            # 将批次转换为张量并转移到目标设备
            batch_tensor = torch.stack(batch_images).to(device)
            batch_tensors.append(batch_tensor)
            
            # 清理临时变量释放内存
            del batch_images
            torch.cuda.empty_cache() if PRELOAD_TO_GPU else None
        
        self.data_tensor = torch.cat(batch_tensors, dim=0)
        self.label_tensor = torch.tensor(self.labels).to(device)
        
        print("预加载完成!")
    
    def to_h5(self, h5_path, compression="gzip", compression_opts=9):
        print(f"将数据集保存到HDF5文件: {h5_path}")
        
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', 
                            data=self.data_tensor.cpu().numpy(), 
                            compression=compression, 
                            compression_opts=compression_opts)
            
            f.create_dataset('labels', 
                            data=self.label_tensor.cpu().numpy(), 
                            compression=compression, 
                            compression_opts=compression_opts)
            
            dt = h5py.special_dtype(vlen=str)
            paths_ds = f.create_dataset('sample_paths', 
                                       (len(self.samples),), 
                                       dtype=dt)
            for i, path in enumerate(self.samples):
                paths_ds[i] = path
            
            label_dict_json = json.dumps(self.label_dict)
            f.attrs['label_dict'] = label_dict_json
            
            f.attrs['num_samples'] = len(self.samples)
            f.attrs['num_classes'] = len(self.label_dict)
            f.attrs['data_shape'] = str(self.data_tensor.shape)
            f.attrs['preloaded_to_gpu'] = PRELOAD_TO_GPU
            f.attrs['preloaded_to_cpu'] = PRELOAD_TO_CPU
        
        print(f"数据集已保存到HDF5文件: {h5_path}")
        self.h5_path = h5_path
    
    def _load_from_h5(self):
        print(f"从HDF5文件加载数据集: {self.h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            data_np = np.array(f['data'])
            labels_np = np.array(f['labels'])
            
            device = torch.device('cuda' if PRELOAD_TO_GPU else 'cpu')
            self.data_tensor = torch.from_numpy(data_np).to(device)
            self.label_tensor = torch.from_numpy(labels_np).to(device)
            
            self.samples = [path.decode('utf-8') if isinstance(path, bytes) else path 
                           for path in f['sample_paths']]
            
            label_dict_json = f.attrs['label_dict']
            if isinstance(label_dict_json, bytes):
                label_dict_json = label_dict_json.decode('utf-8')
            self.label_dict = json.loads(label_dict_json)
            
            print(f"从HDF5加载完成:")
            print(f"  样本数量: {f.attrs['num_samples']}")
            print(f"  类别数量: {f.attrs['num_classes']}")
            print(f"  数据形状: {f.attrs['data_shape']}")
            print(f"  预加载到GPU: {f.attrs.get('preloaded_to_gpu', False)}")
            print(f"  预加载到CPU: {f.attrs.get('preloaded_to_cpu', False)}")
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.data_tensor is not None:
            return self.data_tensor[idx], self.label_tensor[idx], self.samples[idx]
        else:
            img_path = self.samples[idx]
            label = self.labels[idx]
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image, label, img_path
    
    def get_num_classes(self):
        return len(self.label_dict)
    
    def get_h5_info(self):
        if self.h5_path is None or not os.path.exists(self.h5_path):
            return None
        
        info = {}
        with h5py.File(self.h5_path, 'r') as f:
            info['file_size'] = os.path.getsize(self.h5_path) / (1024**3)  # GB
            info['num_samples'] = f.attrs.get('num_samples', 0)
            info['num_classes'] = f.attrs.get('num_classes', 0)
            info['data_shape'] = f.attrs.get('data_shape', 'Unknown')
            info['compression'] = f['data'].compression
            info['compression_opts'] = f['data'].compression_opts
        
        return info
    
if __name__ == '__main__':
    # test
    dataset = PreloadedDataset(root_dir=DATA_ROOT, transform=get_train_transform())
    print(f"数据集大小: {len(dataset)}")
    print(f"类别数量: {dataset.get_num_classes()}")
    
    # 保存到HDF5
    dataset.to_h5('dataset.h5')
    
    # 从HDF5加载
    loaded_dataset = PreloadedDataset(h5_path='dataset.h5')
    print(f"从HDF5加载的数据集大小: {len(loaded_dataset)}")
    print(f"从HDF5加载的类别数量: {loaded_dataset.get_num_classes()}")