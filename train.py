import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2 as cv

import densetorch as dt
from config import *

# === Configuração do modelo ===
#MobileNetV2
# pretrained = False
# return_idx = [1, 2, 3, 4, 5, 6]
# collapse_ind = [[0, 1], [2, 3], 4, 5]

#ResNet50
pretrained = False
return_idx = [0, 1, 2, 3]  # conv2_x a conv5_x
collapse_ind = [[0], [1], [2], [3]]  # um nível por saída

dt.misc.set_seed(seed)
writer = SummaryWriter(log_dir='runs/experiment_ResNet101_0.2_0.8')

# === Dados ===
transform_common = [dt.data.Normalise(*normalise_params), dt.data.ToTensor()]
transform_train = transforms.Compose(
    [dt.data.RandomMirror(), dt.data.RandomCrop(crop_size)] + transform_common
)
transform_val = transforms.Compose(transform_common)

trainloader = DataLoader(
    dt.data.MMDataset(data_file, data_dir, line_to_paths_fn, masks_names, transform=transform_train),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
)
valloader = DataLoader(
    dt.data.MMDataset(val_file, data_val_dir, line_to_paths_fn, masks_names, transform=transform_val),
    batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False,
)

# === Modelo ===
enc = dt.nn.resnet.resnet101(pretrained=pretrained, return_idx=return_idx)
dec = dt.nn.MTLWRefineNet(enc._out_c, collapse_ind, num_classes)
model1 = nn.DataParallel(nn.Sequential(enc, dec).cuda())
print("Model has {} parameters".format(dt.misc.compute_params(model1)))

start_epoch, _, state_dict = saver.maybe_load(
    ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"],
)
dt.misc.load_state_dict(model1, state_dict)
if start_epoch is None:
    start_epoch = 0

# === Otimizadores e schedulers ===
optims = [
    dt.misc.create_optim(optim_enc, enc.parameters(), lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc),
    dt.misc.create_optim(optim_dec, dec.parameters(), lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec),
]

# Um scheduler para cada otimizador
opt_scheds = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, verbose=True)for opt in optims]

print(f"Otimizador encoder: {type(optims[0]).__name__}")
print(f"Otimizador decoder: {type(optims[1]).__name__}")

# === Métricas ===
metric_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_f1_per_class = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes[0], ignore_index=ignore_index, average=None).cuda()

print(f"Coeffs: {loss_coeffs[0]}, {loss_coeffs[1]}")

for i in range(start_epoch, n_epochs):
    model1.train()
    print("Epoch {:d}".format(i))

    # Reset métricas
    metric_accuracy.reset()
    metric_precision.reset()
    metric_recall.reset()
    metric_f1.reset()
    metric_f1_per_class.reset()
    miou_metric = dt.engine.MeanIoU(num_classes[0])
    rmse_metric = dt.engine.RMSE(ignore_val=ignore_depth)

    train_loss_total = 0.0
    n_batches = 0

    for batch in tqdm(trainloader, desc=f"Train Epoch {i}"):
        inputs = batch["image"].float().cuda()
        segm_gt = batch["segm"].cuda()
        depth_gt = batch["depth"].cuda()

        for opt in optims:
            opt.zero_grad()

        outputs = model1(inputs)
        segm_out, depth_out = outputs

        segm_gt_resized = torch.nn.functional.interpolate(
            segm_gt.unsqueeze(1).float(), size=segm_out.shape[2:], mode="nearest").squeeze(1).long()
        depth_gt_resized = torch.nn.functional.interpolate(
            depth_gt.unsqueeze(1), size=depth_out.shape[2:], mode="bilinear", align_corners=False).squeeze(1)

        loss_segm = crit_segm(segm_out, segm_gt_resized)
        loss_depth = crit_depth(depth_out.squeeze(1), depth_gt_resized)
        loss = loss_coeffs[0] * loss_segm + loss_coeffs[1] * loss_depth
        loss.backward()

        for opt in optims:
            opt.step()

        train_loss_total += loss.item()
        n_batches += 1

        preds = torch.argmax(segm_out, dim=1)

        # Atualizar métricas
        metric_accuracy.update(preds, segm_gt_resized)
        metric_precision.update(preds, segm_gt_resized)
        metric_recall.update(preds, segm_gt_resized)
        metric_f1.update(preds, segm_gt_resized)
        metric_f1_per_class.update(preds, segm_gt_resized)
        miou_metric.update(segm_out.detach().cpu().numpy(), segm_gt_resized.cpu().numpy())
        rmse_metric.update(depth_out.squeeze(1).detach().cpu().numpy(), depth_gt_resized.cpu().numpy())

        # ====== Aqui entra o plot do depth para a primeira imagem do primeiro batch de cada época ======
        if n_batches == 1:
            # Visualização do depth (GT e predição)
            gt_depth_img = depth_gt_resized[0].detach().cpu().numpy()
            pred_depth_img = depth_out.squeeze(1)[0].detach().cpu().numpy()

            # Plot com matplotlib
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(gt_depth_img, cmap='plasma')
            axs[0].set_title('GT Depth')
            axs[0].axis('off')

            axs[1].imshow(pred_depth_img, cmap='plasma')
            axs[1].set_title('Predicted Depth')
            axs[1].axis('off')

            plt.tight_layout()
            #plt.savefig(f"depth_vis_train_epoch_{i}.png")  # Salva em arquivo (evita travar em headless)
            plt.close()

            # Adiciona ao TensorBoard (opcional)
            writer.add_image("Train/Depth_GT", torch.tensor(gt_depth_img).unsqueeze(0), i, dataformats='CHW')
            writer.add_image("Train/Depth_Pred", torch.tensor(pred_depth_img).unsqueeze(0), i, dataformats='CHW')

    mean_train_loss = train_loss_total / n_batches
    train_acc = metric_accuracy.compute().item()
    train_prec = metric_precision.compute().item()
    train_rec = metric_recall.compute().item()
    train_f1 = metric_f1.compute().item()
    train_miou = miou_metric.val()
    train_rmse = rmse_metric.val()

    # Log no TensorBoard
    writer.add_scalar("Loss/train", mean_train_loss, i)
    writer.add_scalar("Accuracy/train", train_acc, i)
    writer.add_scalar("Precision/train", train_prec, i)
    writer.add_scalar("Recall/train", train_rec, i)
    writer.add_scalar("F1/train", train_f1, i)
    writer.add_scalar("mIoU/train", train_miou, i)
    writer.add_scalar("RMSE/train", train_rmse, i)

    if i % val_every == 0:
        model1.eval()
        miou_metric = dt.engine.MeanIoU(num_classes[0])
        rmse_metric = dt.engine.RMSE(ignore_val=ignore_depth)

        metric_accuracy.reset()
        metric_precision.reset()
        metric_recall.reset()
        metric_f1.reset()
        metric_f1_per_class.reset()

        val_loss_total = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        #os.makedirs(f"predictions_epoch_{i}", exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valloader, desc=f"Val Epoch {i}")):
                inputs = batch["image"].float().cuda()
                segm_gt = batch["segm"].cuda()
                depth_gt = batch["depth"].cuda()

                outputs = model1(inputs)
                segm_out, depth_out = outputs

                segm_gt_resized = torch.nn.functional.interpolate(
                    segm_gt.unsqueeze(1).float(), size=segm_out.shape[2:], mode="nearest").squeeze(1).long()
                depth_gt_resized = torch.nn.functional.interpolate(
                    depth_gt.unsqueeze(1), size=depth_out.shape[2:], mode="bilinear", align_corners=False).squeeze(1)

                loss_segm = crit_segm(segm_out, segm_gt_resized)
                loss_depth = crit_depth(depth_out.squeeze(1), depth_gt_resized)
                loss = loss_coeffs[0] * loss_segm + loss_coeffs[1] * loss_depth
                val_loss_total += loss.item()
                n_batches += 1

                preds = torch.argmax(segm_out, dim=1)

                metric_accuracy.update(preds, segm_gt_resized)
                metric_precision.update(preds, segm_gt_resized)
                metric_recall.update(preds, segm_gt_resized)
                metric_f1.update(preds, segm_gt_resized)
                metric_f1_per_class.update(preds, segm_gt_resized)

                miou_metric.update(segm_out.detach().cpu().numpy(), segm_gt_resized.cpu().numpy())
                rmse_metric.update(depth_out.squeeze(1).cpu().numpy(), depth_gt_resized.cpu().numpy())

                mask_valid = (segm_gt_resized != ignore_index)
                all_preds.extend(preds[mask_valid].cpu().numpy().flatten())
                all_targets.extend(segm_gt_resized[mask_valid].cpu().numpy().flatten())

                # Salvar predições coloridas
                if batch_idx == 0:
                    writer.add_image("Input/Image", vutils.make_grid(inputs[:4].cpu(), normalize=True), i)
                    segm_gt_vis = torch.stack([p.float() / num_classes[0] for p in segm_gt_resized[:4]]).unsqueeze(1)
                    pred_vis = torch.stack([p.float() / num_classes[0] for p in preds[:4]]).unsqueeze(1)
                    writer.add_image("GT/Segmentation", vutils.make_grid(segm_gt_vis, normalize=True), i)
                    writer.add_image("Pred/Segmentation", vutils.make_grid(pred_vis, normalize=True), i)
                    writer.add_image("GT/Depth", vutils.make_grid(depth_gt_resized[:4].unsqueeze(1), normalize=True), i)
                    writer.add_image("Pred/Depth", vutils.make_grid(depth_out[:4], normalize=True), i)

        mean_loss = val_loss_total / n_batches
        acc = metric_accuracy.compute().item()
        prec = metric_precision.compute().item()
        rec = metric_recall.compute().item()
        f1 = metric_f1.compute().item()
        f1_per_class = metric_f1_per_class.compute().cpu().numpy()
        miou = miou_metric.val()
        rmse = rmse_metric.val()

        # Passar a métrica (loss de validação) para cada scheduler
        for sched in opt_scheds:
            sched.step(mean_loss)


        print(f"\nVal @ Epoch {i}")
        print(f"Loss: {mean_loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"F1 por classe: {np.round(f1_per_class, 4)}")

        writer.add_scalar("Loss/val", mean_loss, i)
        writer.add_scalar("Accuracy/val", acc, i)
        writer.add_scalar("Precision/val", prec, i)
        writer.add_scalar("Recall/val", rec, i)
        writer.add_scalar("F1/val", f1, i)
        writer.add_scalar("mIoU/val", miou, i)
        writer.add_scalar("RMSE/val", rmse, i)

        csv_file = "val_metrics_log.csv"
        csv_exists = os.path.exists(csv_file)
        with open(csv_file, "a") as f:
            writer_csv = csv.writer(f)
            if not csv_exists:
                header = ["epoch", "loss", "accuracy", "precision", "recall", "f1", "mIoU", "RMSE"]
                header += [f"f1_class_{c}" for c in range(num_classes[0])]
                writer_csv.writerow(header)
            row = [i, mean_loss, acc, prec, rec, f1, miou, rmse]
            row += list(f1_per_class)
            writer_csv.writerow(row)

        saver.maybe_save(
            new_val=(miou, rmse),
            dict_to_save={"state_dict": model1.state_dict(), "epoch": i},
        )

# === Encerrar TensorBoard ===
writer.close()

# === Função para aplicar colormap ===
def convert_to_new_colormap(pred_mask, colormap):
    h, w = pred_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        rgb_mask[pred_mask == class_id] = color
    return rgb_mask

# === Colormap usado para as predições ===
NEW_COLORMAP = {
    0: (135, 206, 250),    
    1: (0, 128, 0),    
    2: (107, 142, 35),       
    3: (255, 69, 0),
    4: (139, 0, 0),        
}

# === Função de predição ===
def predict(model, img_dir_val, output_dir, normalise_params):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.35519162, 0.38581184, 0.31314772), std=(0.18028925, 0.16416986, 0.15931557))
    ])

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
            # Corrigido para tratar saída que pode ser lista ou tupla
            segm_out = output[0] if isinstance(output, (tuple, list)) else output

            _, pred_mask = torch.max(segm_out, dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            rgb_mask = convert_to_new_colormap(pred_mask, NEW_COLORMAP)

            output_path = os.path.join(output_dir, os.path.splitext(image_file)[0] + "_prediction_rgb.png")
            cv.imwrite(output_path, cv.cvtColor(rgb_mask, cv.COLOR_RGB2BGR))

    print(f"Predições salvas em {output_dir}")

# === Caminho para imagens de validação ===
img_dir_val = "/home/lucas-alves/dataset_LARS/val/rgb"
output_dir = "/home/lucas-alves/DenseTorch/examples/multitask/predict_ResNet101_0.2_0.8"

# === Fazer predição com o último modelo ===
predict(model1, img_dir_val, output_dir, normalise_params)
