import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchmetrics
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import densetorch as dt
from config import *

# === Configuração do modelo (single-task) ===
pretrained = False
return_idx = [0, 1, 2, 3]  # conv2_x a conv5_x
collapse_ind = [[0], [1], [2], [3]]

dt.misc.set_seed(seed)
writer = SummaryWriter(log_dir='runs/experiment_Resnet50_SingleTask')

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
enc = dt.nn.resnet.resnet50(pretrained=pretrained, return_idx=return_idx)
dec = dt.nn.MTLWRefineNet(enc._out_c, collapse_ind, [num_classes[0]])
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
opt_scheds = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, verbose=True) for opt in optims]

# === Métricas ===
metric_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_precision = torchmetrics.classification.MulticlassPrecision(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_recall = torchmetrics.classification.MulticlassRecall(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes[0], ignore_index=ignore_index).cuda()
metric_f1_per_class = torchmetrics.classification.MulticlassF1Score(num_classes=num_classes[0], ignore_index=ignore_index, average=None).cuda()

# === Loop de treinamento ===
for i in range(start_epoch, n_epochs):
    model1.train()
    print("Epoch {:d}".format(i))

    metric_accuracy.reset()
    metric_precision.reset()
    metric_recall.reset()
    metric_f1.reset()
    metric_f1_per_class.reset()
    miou_metric = dt.engine.MeanIoU(num_classes[0])

    train_loss_total = 0.0
    n_batches = 0

    for batch in tqdm(trainloader, desc=f"Train Epoch {i}"):
        inputs = batch["image"].float().cuda()
        segm_gt = batch["segm"].cuda()

        for opt in optims:
            opt.zero_grad()

        segm_out = model1(inputs)[0]  

        segm_gt_resized = torch.nn.functional.interpolate(
            segm_gt.unsqueeze(1).float(), size=segm_out.shape[2:], mode="nearest").squeeze(1).long()

        loss = crit_segm(segm_out, segm_gt_resized)
        loss.backward()

        for opt in optims:
            opt.step()

        train_loss_total += loss.item()
        n_batches += 1

        preds = torch.argmax(segm_out, dim=1)
        metric_accuracy.update(preds, segm_gt_resized)
        metric_precision.update(preds, segm_gt_resized)
        metric_recall.update(preds, segm_gt_resized)
        metric_f1.update(preds, segm_gt_resized)
        metric_f1_per_class.update(preds, segm_gt_resized)
        miou_metric.update(segm_out.detach().cpu().numpy(), segm_gt_resized.cpu().numpy())

    mean_train_loss = train_loss_total / n_batches
    train_acc = metric_accuracy.compute().item()
    train_prec = metric_precision.compute().item()
    train_rec = metric_recall.compute().item()
    train_f1 = metric_f1.compute().item()
    train_miou = miou_metric.val()

    writer.add_scalar("Loss/train", mean_train_loss, i)
    writer.add_scalar("Accuracy/train", train_acc, i)
    writer.add_scalar("Precision/train", train_prec, i)
    writer.add_scalar("Recall/train", train_rec, i)
    writer.add_scalar("F1/train", train_f1, i)
    writer.add_scalar("mIoU/train", train_miou, i)

    if i % val_every == 0:
        model1.eval()
        miou_metric = dt.engine.MeanIoU(num_classes[0])
        metric_accuracy.reset()
        metric_precision.reset()
        metric_recall.reset()
        metric_f1.reset()
        metric_f1_per_class.reset()

        val_loss_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valloader, desc=f"Val Epoch {i}")):
                inputs = batch["image"].float().cuda()
                segm_gt = batch["segm"].cuda()

                segm_out = model1(inputs)[0] 

                segm_gt_resized = torch.nn.functional.interpolate(
                    segm_gt.unsqueeze(1).float(), size=segm_out.shape[2:], mode="nearest").squeeze(1).long()

                loss = crit_segm(segm_out, segm_gt_resized)
                val_loss_total += loss.item()
                n_batches += 1

                preds = torch.argmax(segm_out, dim=1)
                metric_accuracy.update(preds, segm_gt_resized)
                metric_precision.update(preds, segm_gt_resized)
                metric_recall.update(preds, segm_gt_resized)
                metric_f1.update(preds, segm_gt_resized)
                metric_f1_per_class.update(preds, segm_gt_resized)
                miou_metric.update(segm_out.detach().cpu().numpy(), segm_gt_resized.cpu().numpy())

                if batch_idx == 0:
                    writer.add_image("Input/Image", vutils.make_grid(inputs[:4].cpu(), normalize=True), i)
                    segm_gt_vis = torch.stack([p.float() / num_classes[0] for p in segm_gt_resized[:4]]).unsqueeze(1)
                    pred_vis = torch.stack([p.float() / num_classes[0] for p in preds[:4]]).unsqueeze(1)
                    writer.add_image("GT/Segmentation", vutils.make_grid(segm_gt_vis, normalize=True), i)
                    writer.add_image("Pred/Segmentation", vutils.make_grid(pred_vis, normalize=True), i)

        mean_loss = val_loss_total / n_batches
        acc = metric_accuracy.compute().item()
        prec = metric_precision.compute().item()
        rec = metric_recall.compute().item()
        f1 = metric_f1.compute().item()
        f1_per_class = metric_f1_per_class.compute().cpu().numpy()
        miou = miou_metric.val()

        for sched in opt_scheds:
            sched.step(mean_loss)

        print(f"\nVal @ Epoch {i}")
        print(f"Loss: {mean_loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"mIoU: {miou:.4f}")
        print(f"F1 por classe: {np.round(f1_per_class, 4)}")

        writer.add_scalar("Loss/val", mean_loss, i)
        writer.add_scalar("Accuracy/val", acc, i)
        writer.add_scalar("Precision/val", prec, i)
        writer.add_scalar("Recall/val", rec, i)
        writer.add_scalar("F1/val", f1, i)
        writer.add_scalar("mIoU/val", miou, i)

        csv_file = "val_metrics_log.csv"
        csv_exists = os.path.exists(csv_file)
        with open(csv_file, "a") as f:
            writer_csv = csv.writer(f)
            if not csv_exists:
                header = ["epoch", "loss", "accuracy", "precision", "recall", "f1", "mIoU"]
                header += [f"f1_class_{c}" for c in range(num_classes[0])]
                writer_csv.writerow(header)
            row = [i, mean_loss, acc, prec, rec, f1, miou]
            row += list(f1_per_class)
            writer_csv.writerow(row)

        saver.maybe_save(
            new_val=(miou,),
            dict_to_save={"state_dict": model1.state_dict(), "epoch": i},
        )

writer.close()
