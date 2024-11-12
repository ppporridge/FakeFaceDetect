import os
import json
import csv

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from model import swin_tiny_patch4_window7_224 as create_model
from model import swin_base_patch4_window12_384_in22k as create_model   # 记得改

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 读取class_indices
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(num_classes=2).to(device)   # 记得改

    # 加载模型权重
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # 遍历文件夹中的图片
    img_folder = "/testdata"
    #img_folder = "/root/autodl-tmp/testdata"
    assert os.path.exists(img_folder), f"folder: '{img_folder}' does not exist."

    predictions=[]
    for img_name in os.listdir(img_folder):
        if img_name.endswith(('.jpeg', '.jpg', '.png')):  # 只选择图片文件
            img_path = os.path.join(img_folder, img_name)

            # 加载图片
            img = Image.open(img_path).convert('RGB')
            # [N, C, H, W]
            img = data_transform(img)
            # 扩展batch维度
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # 预测类别
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            # 获取最高概率及对应的类名
            predicted_class = class_indict[str(predict_cla)]
            predicted_prob = predict[predict_cla].numpy()

            predictions.append([os.path.splitext(img_name)[0], predicted_class])
    
    # 按图片名称字符串升序排序
    sorted_predictions = sorted(predictions, key=lambda x: x[0])
    # 定义CSV文件
    output_csv = '../cla_pre.csv'
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted_predictions)

if __name__ == '__main__':
    main()
