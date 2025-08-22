import os
import cv2
import numpy as np
import dlib
from pathlib import Path
import glob

from config.settings import DLIB_FILE, IMAGE_SIZE

def preprocess(input_dir="data", output_dir="preprocesseddata", output_size=(224, 224)):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # dlib关键点预测器
    detector = dlib.get_frontal_face_detector()
    
    predictor_path = DLIB_FILE
    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"关键点预测器模型文件未找到: {predictor_path}")
    
    predictor = dlib.shape_predictor(predictor_path)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    
    #所有人名子目录
    person_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    processed_count = 0
    skipped_count = 0
    
    for person_name in person_dirs:
        person_input_dir = os.path.join(input_dir, person_name)
        person_output_dir = os.path.join(output_dir, person_name)
        Path(person_output_dir).mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(person_input_dir, extension)))
        
        print(f"处理 {person_name} 的 {len(image_paths)} 张图像")
        
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)# 使用imread是因为支持的格式多
                if image is None:
                    print(f"无法读取图像: {image_path}")
                    skipped_count += 1
                    continue
                
                # 转换为灰度图用于人脸检测
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                faces = detector(gray, 1)
                if len(faces) == 0:
                    print(f"未检测到人脸: {image_path}")
                    skipped_count += 1
                    continue
    
                face = faces[0]
                
                #关键点
                landmarks = predictor(gray, face)
                landmarks_points = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points.append((x, y))
                
                aligned_face = align_face(image, landmarks_points, output_size)
                
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(person_output_dir, f"{name}_aligned.jpg")
                
                cv2.imwrite(output_path, aligned_face)
                processed_count += 1
                
            except Exception as e:
                print(f"处理图像时出错 {image_path}: {str(e)}")
                skipped_count += 1
    
    print(f"预处理完成! 成功处理: {processed_count}, 跳过: {skipped_count}")

def align_face(image, landmarks, output_size=(224, 224)):
    # 选择眼睛关键点（左眼36-41，右眼42-47）
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

    # 旋转对齐
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_CUBIC)
    
    landmarks_np = np.array([(x, y, 1) for (x, y) in landmarks])
    rotated_landmarks = np.dot(rotation_matrix, landmarks_np.T).T
    
    min_x = int(np.min(rotated_landmarks[:,0]))
    max_x = int(np.max(rotated_landmarks[:,0]))
    min_y = int(np.min(rotated_landmarks[:,1]))
    max_y = int(np.max(rotated_landmarks[:,1]))
    
    # 扩展裁剪区域以确保包含完整人脸
    width = max_x - min_x
    height = max_y - min_y
    margin_x = width * 0.8
    margin_y = height * 0.8
    
    crop_x1 = max(0, int(min_x - margin_x))
    crop_y1 = max(0, int(min_y - margin_y))
    crop_x2 = min(image.shape[1], int(max_x + margin_x))
    crop_y2 = min(image.shape[0], int(max_y + margin_y))
    
    cropped_face = aligned_face[crop_y1:crop_y2, crop_x1:crop_x2]
    
    resized_face = cv2.resize(cropped_face, output_size)
    
    return resized_face

def preprocessone(input_path, output_path, output_size=(IMAGE_SIZE, IMAGE_SIZE)):
    image = cv2.imread(input_path)
    if image is None:
        print(f"无法读取图像: {input_path}")
        return
    
    # 转换为灰度图用于人脸检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_FILE)
    
    faces = detector(gray, 1)
    if len(faces) == 0:
        print(f"未检测到人脸: {input_path}")
        return
    
    face = faces[0]
    
    # 检测关键点
    landmarks = predictor(gray, face)
    landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
    
    # 对齐和裁剪人脸
    aligned_face = align_face(image, landmarks_points, output_size)
    
    # 保存处理后的图像
    cv2.imwrite(output_path, aligned_face)


if __name__ == "__main__":
    preprocess(input_dir="testdataraw", output_dir="testdata", output_size=(IMAGE_SIZE, IMAGE_SIZE))