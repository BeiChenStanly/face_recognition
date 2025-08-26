# 人脸识别项目
## 摘要
使用预训练的ResNet50，损失函数使用ArcFace，数据集使用Youtube Faces
## 使用导引
1. 下载[Youtube Faces](https://www.cs.tau.ac.il/~wolf/ytfaces/index.html#download)中的aligned_images_DB.tar.gz并解压，接着在config/settings.py中设置数据集路径，你也可以设置其他参数
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
   注意：如果你有GPU并且想使用CUDA版本的PyTorch，请参考[PyTorch官网](https://pytorch.org/get-started/locally/)安装
3. 训练模型
   ```bash
   python train.py
   ```
## 训练完成后自己组织数据库识别
1. 数据集组织
testdataraw/人名/图片
2. 识别测试
   在训练完成后，你可以使用提供的`test.py`脚本进行识别测试：
   ```bash
   python test.py --img /path/to/your/image.*
   ```