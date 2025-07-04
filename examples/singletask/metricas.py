import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import shutil

# Mapeamento de cores para classes (ground truth)
# CUSTOM_COLORMAP = {
#     (1, 1, 1): 0,    # 1: (135, 206, 250)
#     (2, 2, 2): 1,    # 2: (0, 128, 0)
#     (3, 3, 3): 2,    # 3: (107, 142, 35)
#     (6, 6, 6): 3,    # 6: (255, 69, 0)
#     (9, 9, 9): 4     # 9: 
# }

# Mapeamento de cores para classes (predição)
CUSTOM_COLORMAP = {
    (135, 206, 250): 0,  # Unclassified
    (0, 128, 0): 1,  # Ground
    (107, 142, 35): 2,  # Vegetation
    (255, 69, 0): 3,  # Building
    (139, 0, 0) : 4  # Water
}

PREDICTION_COLORMAP = {
    (135, 206, 250): 0,  # Unclassified
    (0, 128, 0): 1,  # Ground
    (107, 142, 35): 2,  # Vegetation
    (255, 69, 0): 3,  # Building
    (139, 0, 0) : 4  # Water
}

DEFAULT_CLASS = -1

# Lista de classes para rótulos das matrizes
target_labels = ["Ground", "Vegetation", "Building", "Water"]

def get_center_crop_coords(h, w, multiple=32):
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    return start_y, start_x, new_h, new_w

def crop_image(img_path, start_y, start_x, new_h, new_w):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {img_path}")
    return img[start_y:start_y + new_h, start_x:start_x + new_w]

def convert_image_to_classes(image_path, colormap, default_class=DEFAULT_CLASS, is_path=True):
    if is_path:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Carrega como BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Converte para RGB        
    else:
        img = image_path
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # <- ESSENCIAL PARA GT

    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_NEAREST)  # <- remova isso se não quiser redimensionar
    class_map = np.full((img.shape[0], img.shape[1]), default_class, dtype=np.int32)

    for rgb, class_id in colormap.items():
        mask = np.all(img == np.array(rgb), axis=-1)
        class_map[mask] = class_id

    return class_map

# Função para salvar matrizes de confusão em porcentagem
def save_confusion_matrix(conf_matrix, filename, title):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_pct = np.divide(conf_matrix, row_sums, where=row_sums != 0)  # Evita divisão por zero
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(conf_matrix_pct, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=target_labels, yticklabels=target_labels)
    plt.xlabel("Predito")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close()

# Função para salvar a máscara de erro e a imagem sobreposta
def save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, image_name):
    if not os.path.exists(original_image_path):
        print(f"Erro: Imagem original não encontrada em {original_image_path}")
        return
    
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    original_image = cv2.resize(original_image, (100, 100), interpolation=cv2.INTER_NEAREST) 
    if original_image is None:
        print(f"Erro: Falha ao carregar a imagem {original_image_path}")
        return

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Aplicar o mesmo crop que foi feito nas segmentações
    #start_y, start_x, new_h, new_w = get_center_crop_coords(original_image.shape[0], original_image.shape[1])
    #original_image = original_image[start_y:start_y + new_h, start_x:start_x + new_w]


    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    #start_y, start_x, new_h, new_w = get_center_crop_coords(gt_img.shape[0], gt_img.shape[1])
    #gt_cropped = crop_image(gt_path, start_y, start_x, new_h, new_w)


    # Aplica crop nas imagens
    #gt_cropped = crop_image(gt_path, start_y, start_x, new_h, new_w)
    #pred_cropped = crop_image(pred_path, start_y, start_x, new_h, new_w)

    gt_classes = convert_image_to_classes(gt_img, CUSTOM_COLORMAP, is_path=False)
    pred_classes = convert_image_to_classes(pred_path, PREDICTION_COLORMAP, is_path=True)    


    # Criando a máscara binária de erro (1 para erro, 0 para acerto)
    valid_mask = (gt_classes != DEFAULT_CLASS) & (pred_classes != DEFAULT_CLASS)
    
    error_mask = np.zeros_like(gt_classes, dtype=np.uint8)
    error_mask[valid_mask] = (gt_classes[valid_mask] != pred_classes[valid_mask]).astype(np.uint8) * 255
    

    cv2.imwrite(os.path.join(output_folder, f"{image_name}_error_mask.png"), error_mask)

    # Criando uma cópia da imagem original para sobreposição
    overlay = original_image.copy()

    # Criando uma imagem colorida da máscara para visualização
    error_colored = np.zeros_like(original_image)
    error_colored[:, :, 0] = error_mask  # Define a máscara como vermelha (canal R)

    # Aplicando sobreposição com transparência apenas nos pixels de erro
    alpha = 0.2  # Grau de transparência
    mask_indices = error_mask > 0  # Índices onde há erro
    overlay[mask_indices] = cv2.addWeighted(original_image[mask_indices], 1 - alpha, error_colored[mask_indices], alpha, 0)

    # Salvando a imagem final com a sobreposição
    cv2.imwrite(os.path.join(output_folder, f"{image_name}_error_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

# Caminho das imagens
gt_folder = '/home/lucas-alves/dataset_LARS/val/labels_colorido'
pred_folder = "/home/lucas-alves/DenseTorch/examples/singletask/predict_ResNet50_single"
output_folder = "/home/lucas-alves/DenseTorch/examples/singletask/matriz_ResNet50_single"
image_folder = '/home/lucas-alves/dataset_LARS/val/rgb' # Pasta de imagens originais

os.makedirs(output_folder, exist_ok=True)

# Lista de imagens
image_files = [f for f in os.listdir(gt_folder) if f.endswith(".png")]

# Imagens específicas
specific_images = [
]

# Matriz de confusão total
total_conf_matrix = np.zeros((4, 4), dtype=int)

scores = []
matrices = {}

for image_file in image_files:
    gt_path = os.path.join(gt_folder, image_file)
    pred_path = os.path.join(pred_folder, image_file.replace(".png", "_prediction_rgb.png"))
    
    if not os.path.exists(pred_path):
        continue
    
    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    #start_y, start_x, new_h, new_w = get_center_crop_coords(gt_img.shape[0], gt_img.shape[1])
    #gt_cropped = crop_image(gt_path, start_y, start_x, new_h, new_w)

    gt_classes = convert_image_to_classes(gt_img, CUSTOM_COLORMAP, is_path=False)
    pred_classes = convert_image_to_classes(pred_path, PREDICTION_COLORMAP)
    
    # Remover pixels desconhecidos antes da matriz de confusão
    valid_mask = (gt_classes != DEFAULT_CLASS) & (pred_classes != DEFAULT_CLASS)
    conf_matrix = confusion_matrix(gt_classes[valid_mask].ravel(), pred_classes[valid_mask].ravel(), labels=[1, 2, 3, 4])
    
    total_conf_matrix += conf_matrix
    
    score = np.trace(conf_matrix)
    scores.append((score, image_file, conf_matrix))
    matrices[image_file] = conf_matrix

# Salvar a matriz de confusão total
save_confusion_matrix(total_conf_matrix, "total_confusion_matrix.png", "Matriz de Confusão Total")

print("Matriz de confusão total gerada e salva com sucesso!")   

# Gerar matriz de confusão para imagens específicas e salvar erro e overlay
for gt_name, pred_name in specific_images:
    gt_path = os.path.join(gt_folder, gt_name + ".png")
    pred_path = os.path.join(pred_folder, pred_name + ".png")
    original_image_path = os.path.join(image_folder, gt_name + ".png")  # Agora buscando a imagem original
    
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        #start_y, start_x, new_h, new_w = get_center_crop_coords(gt_img.shape[0], gt_img.shape[1])
        #gt_cropped = crop_image(gt_path, start_y, start_x, new_h, new_w)
        gt_classes = convert_image_to_classes(gt_img, CUSTOM_COLORMAP, is_path=False)
        pred_classes = convert_image_to_classes(pred_path, PREDICTION_COLORMAP)
        valid_mask = (gt_classes != DEFAULT_CLASS) & (pred_classes != DEFAULT_CLASS)
        conf_matrix = confusion_matrix(gt_classes[valid_mask].ravel(), pred_classes[valid_mask].ravel(), labels=[1, 2, 3, 4])
        save_confusion_matrix(conf_matrix, f"{gt_name}_confusion_matrix.png", f"Matriz - {gt_name}")
        
        shutil.copy(gt_path, os.path.join(output_folder, f"{gt_name}_ground_truth.png"))
        shutil.copy(pred_path, os.path.join(output_folder, f"{gt_name}_prediction.png"))
        
        # Salvar a máscara de erro e a imagem sobreposta
        save_error_mask_and_overlay(gt_path, pred_path, original_image_path, output_folder, gt_name)
        print(f"Matriz, máscara de erro e sobreposição para {gt_name} salvas.")

# ======= Cálculo de métricas =======
print("\n==== Métricas por classe ====")

TP = np.diag(total_conf_matrix)
FP = total_conf_matrix.sum(axis=0) - TP
FN = total_conf_matrix.sum(axis=1) - TP
TN = total_conf_matrix.sum() - (TP + FP + FN)

precision = np.divide(TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
recall = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
iou = np.divide(TP, TP + FP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FP + FN) != 0)
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)
accuracy = np.divide(TP + TN, total_conf_matrix.sum(), out=np.zeros_like(TP, dtype=float))

for i, label in enumerate(target_labels):
    print(f"\nClasse: {label}")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  IoU:       {iou[i]:.4f}")
    print(f"  F1-score:  {f1[i]:.4f}")
    print(f"  Accuracy:  {accuracy[i]:.4f}")

print("\n==== Métricas médias ====")
print(f"Mean Precision: {np.mean(precision):.4f}")
print(f"Mean Recall:    {np.mean(recall):.4f}")
print(f"Mean IoU:       {np.mean(iou):.4f}")
print(f"Mean F1-score:  {np.mean(f1):.4f}")
print(f"Mean Accuracy:  {np.mean(accuracy):.4f}")