seed = 42
device = "cuda"
start_epoch = 0
stop_epoch = 0
max_epoch_num = 105
warmup = False
warmup_period = 250
dice_param = 0.8
train_batch_size = 2
valid_batch_size = 1

learning_rate = 1e-4
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
    "vit_b": "/root/autodl-tmp/SAM-Adapter-main/weights/sam_pretrain/sam_vit_b_01ec64.pth",
    "vit_l": "/root/autodl-tmp/SAM-Adapter-main/weights/sam_pretrain/sam_vit_l_0b3195.pth",
    "vit_h": "/root/autodl-tmp/SAM-Adapter-main/weights/sam_pretrain/sam_vit_h_4b8939.pth",
}
adapters_weights_paths_dict = {
    "vit_b": None,
    "vit_l": None,
    "vit_h": None,
}

name = "adapter_urban"
model_type = "vit_b"
root_path = "/root/autodl-tmp/outputs_loveda_urban"
vit_dim = vit_dims_dict[model_type]
sam_weights_path = sam_weights_paths_dict[model_type]
adapters_weights_path = adapters_weights_paths_dict[model_type]
checkpoint_name = f"sam_{model_type}_{name}_epoch_score.pth"
output_path = f"{root_path}/weights/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"
logs_path = f"{root_path}/train_logs/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"
tensorboard_path = f"{root_path}/tensorboard/sam-{model_type.replace('_', '-')}-{name.replace('_', '-')}"

### --------------- Configuring the Train and Valid datasets ---------------

dataset_Potsdam = {
    "name": "Potsdam",
    "img_dir": "/root/autodl-tmp/data/potsdam/train/images_1024",
    "gt_dir": "/root/autodl-tmp/data/potsdam/train/masks_1024",
    "img_ext": ".tif",
    "gt_ext": ".png",
}

dataset_Potsdam_val = {
    "name": "Potsdam",
    "img_dir": "/root/autodl-tmp/data/potsdam/test/images_1024",
    "gt_dir": "/root/autodl-tmp/data/potsdam/test/masks_1024",
    "img_ext": ".tif",
    "gt_ext": ".png",
}

dataset_Vaihingen = {
    "name": "Vaihingen",
    "img_dir": "/root/autodl-tmp/data/vaihingen/train/images_1024",
    "gt_dir": "/root/autodl-tmp/data/vaihingen/train/masks_1024",
    "img_ext": ".tif",
    "gt_ext": ".png",
}

dataset_Vaihingen_val = {
    "name": "Vaihingen",
    "img_dir": "/root/autodl-tmp/data/vaihingen/test/images_1024",
    "gt_dir": "/root/autodl-tmp/data/vaihingen/test/masks_1024",
    "img_ext": ".tif",
    "gt_ext": ".png",
}

dataset_Seaice = {
    "name": "Seaice",
    "img_dir": "/root/autodl-tmp/data/seaice/train_images",
    "gt_dir": "/root/autodl-tmp/data/seaice/train_masks",
    "img_ext": ".tif",
    "gt_ext": ".png",
}

dataset_Seaice_val = {
    "name": "Seaice",
    "img_dir": "/root/autodl-tmp/data/seaice/test_images",
    "gt_dir": "/root/autodl-tmp/data/seaice/test_masks",
    "img_ext": ".tif",
    "gt_ext": ".png",
}

dataset_Rural = {
    "name": "Rural",
    "img_dir": "/root/autodl-tmp/data/LoveDA/Train/Rural/images_png",
    "gt_dir": "/root/autodl-tmp/data/LoveDA/Train/Rural/masks_png_convert",
    "img_ext": ".png",
    "gt_ext": ".png",
}

dataset_Rural_val = {
    "name": "Rural",
    "img_dir": "/root/autodl-tmp/data/LoveDA/Val/Rural/images_png",
    "gt_dir": "/root/autodl-tmp/data/LoveDA/Val/Rural/masks_png_convert",
    "img_ext": ".png",
    "gt_ext": ".png",
}

dataset_Urban = {
    "name": "Urban",
    "img_dir": "/root/autodl-tmp/data/LoveDA/Train/Urban/images_png",
    "gt_dir": "/root/autodl-tmp/data/LoveDA/Train/Urban/masks_png_convert",
    "img_ext": ".png",
    "gt_ext": ".png",
}

dataset_Urban_val = {
    "name": "Urban",
    "img_dir": "/root/autodl-tmp/data/LoveDA/Val/Urban/images_png",
    "gt_dir": "/root/autodl-tmp/data/LoveDA/Val/Urban/masks_png_convert",
    "img_ext": ".png",
    "gt_ext": ".png",
}

dataset_water_seg = {
    "name": "ISPRS-Water",
    "img_dir": "/root/autodl-tmp/data/water_seg/train_images",
    "gt_dir": "/root/autodl-tmp/data/water_seg/train_masks",
    "img_ext": ".png",
    "gt_ext": ".png",
}

dataset_water_seg_val = {
    "name": "ISPRS-Water",
    "img_dir": "/root/autodl-tmp/data/water_seg/test_images",
    "gt_dir": "/root/autodl-tmp/data/water_seg/test_masks",
    "img_ext": ".png",
    "gt_ext": ".png",
}

num_classes = 8
classes = ('background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural','no-data')
train_datasets = [dataset_Urban]
valid_datasets = [dataset_Urban_val]
