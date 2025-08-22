import torchvision.transforms as transforms
from config.settings import IMAGE_SIZE

def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
