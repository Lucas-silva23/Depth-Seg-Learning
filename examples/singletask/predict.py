import os
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from config import normalise_params, num_classes
import densetorch as dt

# === Colormap usado para as predições ===
NEW_COLORMAP = {
    0: (135, 206, 250),    
    1: (0, 128, 0),    
    2: (107, 142, 35),       
    3: (255, 69, 0),
    4: (139, 0, 0),        
}

# === Função para aplicar colormap ===
def convert_to_new_colormap(pred_mask, colormap):
    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        rgb_mask[pred_mask == class_id] = color
    return rgb_mask

# === Caminhos ===
ckpt_path = "/home/lucas-alves/DenseTorch/examples/singletask/checkpoint.pth_resnet50_single.tar"
img_dir_val = "/home/lucas-alves/dataset_LARS/val/rgb"
output_dir = "/home/lucas-alves/DenseTorch/examples/singletask/predict_ResNet50_single"

# === Modelo ===
return_idx = [0, 1, 2, 3]
collapse_ind = [[0], [1], [2], [3]]

enc = dt.nn.resnet.resnet50(pretrained=False, return_idx=return_idx)
dec = dt.nn.MTLWRefineNet(enc._out_c, collapse_ind, num_classes)
model = torch.nn.DataParallel(torch.nn.Sequential(enc, dec).cuda())

# === Carregar pesos salvos ===
# === Carregar pesos salvos ===
checkpoint = torch.load(ckpt_path, map_location="cuda")
state_dict = checkpoint["state_dict"]
dt.misc.load_state_dict(model, state_dict)
model.eval()

# === Transformação de entrada ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.35519162, 0.38581184, 0.31314772), std=(0.18028925, 0.16416986, 0.15931557))
])

# === Executar predições ===
os.makedirs(output_dir, exist_ok=True)
image_files = sorted([
    f for f in os.listdir(img_dir_val)
    if os.path.isfile(os.path.join(img_dir_val, f))
])

with torch.no_grad():
    for image_file in tqdm(image_files, desc="Predição"):
        image_path = os.path.join(img_dir_val, image_file)

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (400, 400), interpolation=cv.INTER_NEAREST)

        input_tensor = transform(image).unsqueeze(0).to("cuda")
        output = model(input_tensor)
        segm_out = output[0] if isinstance(output, (tuple, list)) else output

        _, pred_mask = torch.max(segm_out, dim=1)
        pred_mask = pred_mask.squeeze(0).cpu().numpy()

        rgb_mask = convert_to_new_colormap(pred_mask, NEW_COLORMAP)
        output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_prediction_rgb.png")
        cv.imwrite(output_path, cv.cvtColor(rgb_mask, cv.COLOR_RGB2BGR))

print(f"\n✅ Predições salvas em {output_dir}")
