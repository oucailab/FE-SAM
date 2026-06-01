seed = 42
device = "cpu"
start_epoch = 0
stop_epoch = 0
max_epoch_num = 50
warmup = False
warmup_period = 250
dice_param = 0.8
train_batch_size = 2
valid_batch_size = 1

learning_rate = 1e-3
lr_drop_epoch = 10
adam_betas = (0.9, 0.999)
adam_eps = 1e-08
weight_decay = 2.5e-4
sgd_momentum = 0.9

image_size = (1024, 1024)
model_save_fre = 5
deterministic = True
is_eval = False

vit_dims_dict={
    'vit_b': 768,
    'vit_l': 1024,
    'vit_h': 1280,
}
sam_weights_paths_dict = {
    "vit_b": "/root/SAM-Adapter/weights/sam_pretrain/sam_vit_b_01ec64.pth",
    "vit_l": "/root/SAM-Adapter/weights/sam_pretrain/sam_vit_l_0b3195.pth",
    "vit_h": "/root/SAM-Adapter/weights/sam_pretrain/sam_vit_h_4b8939.pth",
}
adapters_weights_paths_dict = {
    "vit_b": None,
    "vit_l": None,
    "vit_h": None,
}

name = "adapter_pd"
model_type = "vit_b"
root_path = "/root/autodl-tmp/outputs"
vit_dim = vit_dims_dict[model_type]
sam_weights_path = sam_weights_paths_dict[model_type]
adapters_weights_path = adapters_weights_paths_dict[model_type]
checkpoint_name = f"sam_{model_type}_{name}_epoch_score.pth"
output_path = f"{root_path}/weights/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"
logs_path = f"{root_path}/train_logs/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"
tensorboard_path = f"{root_path}/tensorboard/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"

### --------------- Configuring the Train and Valid datasets ---------------

dataset_SUIM = {
    "name": "SUIM",
    "img_dir": "/root/autodl-tmp/SUIM/train/images",
    "gt_dir": "/root/autodl-tmp/SUIM/train/labels",
    "img_ext": ".jpg",
    "gt_ext": ".bmp",
}

dataset_SUIM_val = {
    "name": "SUIM",
    "img_dir": "/root/autodl-tmp/SUIM/test/images",
    "gt_dir": "/root/autodl-tmp/SUIM/test/labels",
    "img_ext": ".jpg",
    "gt_ext": ".bmp",
}

num_classes = 8
classes = ('BW', 'HD', 'PF', 'WR', 'RO', 'RI', 'FV', 'SR')
train_datasets = [dataset_SUIM]
valid_datasets = [dataset_SUIM_val]
