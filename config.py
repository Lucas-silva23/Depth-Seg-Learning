import math
import operator
import numpy as np
import torch
import torch.nn as nn
import densetorch as dt

# --------------------
# Random seed
seed = 42

# --------------------
# Dados
crop_size = 400
batch_size = 32
val_batch_size = 32
num_classes = (5, 1)
n_epochs = 100
val_every = 1

data_file = "./lists/train_list_depth.txt"
val_file = "./lists/val_list_depth.txt"
data_dir = "./datasets/Dataset_LARS/"
data_val_dir = "./datasets/Dataset_LARS/"
masks_names = ("segm", "depth")

def line_to_paths_fn(x):
    return x.decode("utf-8").strip("\n").split("\t")

depth_scale = 5000.0
img_scale = 1.0 / 255
img_mean = np.array([0.35519162, 0.38581184, 0.31314772])
img_std = np.array([0.18028925, 0.16416986, 0.15931557])
normalise_params = [
    img_scale,
    img_mean.reshape((1, 1, 3)),
    img_std.reshape((1, 1, 3)),
    depth_scale,
]
ignore_index = 255
ignore_depth = 0

# --------------------
# Cálculo dos pesos para as classes

class_frequencies = {
    0: 0.0158,
    1: 0.5229,
    2: 0.2614,
    3: 0.1024,
    4: 0.0975
}

# --------------------
# Definição das perdas

crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
crit_depth = dt.engine.InvHuberLoss(ignore_index=ignore_depth).cuda()

# --------------------
# Otimizadores e parâmetros

lr_enc = 1e-4
optim_enc = "adam"
mom_enc = 0.0
wd_enc = 0.0

lr_dec = 1e-4
optim_dec = "adam"
mom_dec = 0.0
wd_dec = 0.0

loss_coeffs = (0.2, 0.8)

# --------------------
# Salvamento de checkpoints

init_vals = (0.0, 10000.0)
comp_fns = [operator.gt, operator.lt]
ckpt_dir = "./"
ckpt_path = "./checkpoint.pth.tar"

saver = dt.misc.Saver(
    args=locals(),
    ckpt_dir=ckpt_dir,
    best_val=init_vals,
    condition=comp_fns,
    save_several_mode=all,
)
