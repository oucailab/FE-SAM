import os
import random
import torch
import numpy as np
import ttach as tta
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from importlib import import_module
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
from pathlib import Path
import cv2

from utils.py2cfg import py2cfg
from utils.metric import Evaluator
from utils.dataloader import SegDataset
from utils.dataloader import get_image_mask_mapping
from utils.transforms import data_augmentation as data_aug
from models.segment_anything import sam_model_registry

from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb_potsdam(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]

    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # ImSurf - white
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]      # Building - red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 255]    # LowVeg - yellow
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]      # Tree - green
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 204, 0]    # Car - cyan-blue
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]      # Clutter - blue

    return mask_rgb


def img_writer_rgb(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_png = mask_id + '.png'
        mask_rgb = label2rgb_potsdam(mask)
        cv2.imwrite(mask_name_png, cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def extract_original_filename_from_batch(batch, batch_idx, sample_idx, dataset):
    try:
        absolute_idx = batch_idx * len(batch['image']) + sample_idx

        if hasattr(dataset, 'dataset') and 'img_path' in dataset.dataset:
            img_path = dataset.dataset['img_path'][absolute_idx]
            filename = os.path.splitext(os.path.basename(img_path))[0]
            return filename

        if hasattr(dataset, 'image_list'):
            img_path = dataset.image_list[absolute_idx]
            filename = os.path.splitext(os.path.basename(img_path))[0]
            return filename

        idx_value = batch['idx'][sample_idx]
        if isinstance(idx_value, torch.Tensor):
            if idx_value.dim() == 0:
                idx_str = str(idx_value.item())
            else:
                idx_str = str(idx_value.cpu().numpy())
        else:
            idx_str = str(idx_value)

        if '/' in idx_str or '\\' in idx_str:
            filename = os.path.splitext(os.path.basename(idx_str))[0]
        else:
            filename = idx_str.replace('[', '').replace(']', '').replace(' ', '_').strip()

        return filename

    except Exception as e:
        print(f"Error extracting filename for batch {batch_idx}, sample {sample_idx}: {e}")
        absolute_idx = batch_idx * len(batch['image']) + sample_idx
        return f"sample_{absolute_idx:06d}"


def test_model_with_tta(config, model, test_dataloaders, multimask_output, output_path, tta_type=None, rgb=True):
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    if tta_type == "lr":
        print("Using TTA: Horizontal and Vertical Flip")
    elif tta_type == "d4":
        print("Using TTA: D4 augmentations")
    else:
        print("No TTA applied")

    print(f"Testing on {len(test_dataloaders.dataset)} samples...")
    iterator = tqdm(test_dataloaders, desc="Testing")

    results = []
    total_processed = 0
    successful_saves = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            idx, images, labels = batch['idx'], batch['image'], batch['label']

            for i in range(images.size(0)):
                single_image = images[i:i+1]
                single_label = labels[i:i+1] if labels is not None else None

                original_filename = extract_original_filename_from_batch(
                    batch, batch_idx, i, test_dataloaders.dataset
                )

                if total_processed % 100 == 0:
                    print(f"\nProcessing sample {total_processed}: {original_filename}")
                    print(f"Batch idx: {batch_idx}, Sample idx: {i}")
                    print(f"Image shape: {single_image.shape}")

                if torch.cuda.is_available() and config.device != "cpu":
                    single_image = single_image.cuda()

                try:
                    if tta_type == "lr":
                        predictions_list = []

                        pred, _ = model(single_image, multimask_output, config.image_size[0])
                        predictions_list.append(torch.nn.Softmax(dim=1)(pred["masks"]))

                        img_hflip = torch.flip(single_image, [3])
                        pred_hflip, _ = model(img_hflip, multimask_output, config.image_size[0])
                        pred_hflip = torch.flip(torch.nn.Softmax(dim=1)(pred_hflip["masks"]), [3])
                        predictions_list.append(pred_hflip)

                        predictions = torch.mean(torch.stack(predictions_list), dim=0)
                    elif tta_type == "d4":
                        predictions_list = []

                        pred, _ = model(single_image, multimask_output, config.image_size[0])
                        predictions_list.append(torch.nn.Softmax(dim=1)(pred["masks"]))

                        img_hflip = torch.flip(single_image, [3])
                        pred_hflip, _ = model(img_hflip, multimask_output, config.image_size[0])
                        pred_hflip = torch.flip(torch.nn.Softmax(dim=1)(pred_hflip["masks"]), [3])
                        predictions_list.append(pred_hflip)

                        img_vflip = torch.flip(single_image, [2])
                        pred_vflip, _ = model(img_vflip, multimask_output, config.image_size[0])
                        pred_vflip = torch.flip(torch.nn.Softmax(dim=1)(pred_vflip["masks"]), [2])
                        predictions_list.append(pred_vflip)

                        predictions = torch.mean(torch.stack(predictions_list), dim=0)
                    else:
                        pred, _ = model(single_image, multimask_output, config.image_size[0])
                        predictions = torch.nn.Softmax(dim=1)(pred["masks"])

                    mask = predictions.argmax(dim=1)[0].cpu().numpy()

                    if single_label is not None:
                        label = single_label[0].cpu().numpy()
                        evaluator.add_batch(pre_image=mask, gt_image=label)

                    mask_path = str(output_path / original_filename)
                    results.append((mask, mask_path, rgb))
                    successful_saves += 1

                    del single_image, predictions, mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    print(f"Error processing sample {original_filename}: {e}")

                total_processed += 1

            del images, labels, idx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            iterator.set_postfix({
                'processed': total_processed,
                'successful': successful_saves,
                'current_batch': batch_idx + 1
            })

    print(f"\nTotal samples processed: {total_processed}")
    print(f"Successful predictions: {successful_saves}")
    print(f"Failed predictions: {total_processed - successful_saves}")

    if hasattr(evaluator, 'confusion_matrix') and evaluator.confusion_matrix.sum() > 0:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()

        print("\n=== Evaluation Results ===")
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print(f'F1_{class_name}:{class_f1:.4f}, IOU_{class_name}:{class_iou:.4f}')
        print(f'F1:{np.nanmean(f1_per_class[:-1]):.4f}, mIOU:{np.nanmean(iou_per_class[:-1]):.4f}, OA:{OA:.4f}')

    print(f"\nSaving {len(results)} prediction images...")
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer_rgb, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print(f'Images writing completed in: {img_write_time:.2f} seconds')
    print(f"Results saved to: {output_path}")

    return successful_saves


def get_args():
    parser = argparse.ArgumentParser(description='SAM-Adapter Potsdam Model Testing')
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config file")
    arg("-m", "--model_path", type=Path, required=True, help="Path to trained model weights")
    arg("-o", "--output_path", type=Path, required=True, help="Path where to save resulting masks")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    arg("--rgb", action='store_true', help="Whether to output RGB images")
    arg("--batch_size", type=int, default=1, help="Batch size for testing")
    return parser.parse_args()


def main():
    args = get_args()

    seed_everything(42)

    args.output_path.mkdir(exist_ok=True, parents=True)

    config = py2cfg(args.config_path)

    print("=== SAM-Adapter Potsdam Model Testing ===")
    print(f"Config: {args.config_path}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_path}")
    print(f"TTA: {args.tta}")
    print(f"RGB Output: {args.rgb}")
    print(f"Batch Size: {args.batch_size}")

    cudnn.benchmark = not config.deterministic
    cudnn.deterministic = config.deterministic

    sam = sam_model_registry[config.model_type](
        num_classes=config.num_classes,
        checkpoint=config.sam_weights_path,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
        image_size=config.image_size[0],
    )

    pkg = import_module("models.fesam")
    net = pkg.FESAM(sam).to(config.device)

    multimask_output = config.num_classes > 2

    if args.model_path.exists():
        print(f"Loading model weights from: {args.model_path}")
        try:
            print("Initializing model components...")
            dummy_input = torch.randn(1, 3, config.image_size[0], config.image_size[0]).to(config.device)
            with torch.no_grad():
                _ = net(dummy_input, multimask_output, config.image_size[0])

            print("Loading state dict...")
            state_dict = torch.load(args.model_path, map_location='cpu')

            model_keys = set(net.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            missing_keys = checkpoint_keys - model_keys
            unexpected_keys = model_keys - checkpoint_keys

            if missing_keys:
                print(f"Missing keys in model: {len(missing_keys)} keys")
                for i, key in enumerate(list(missing_keys)[:3]):
                    print(f"  Missing: {key}")
                if len(missing_keys) > 3:
                    print(f"  ... and {len(missing_keys) - 3} more")

            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
                for i, key in enumerate(list(unexpected_keys)[:3]):
                    print(f"  Unexpected: {key}")
                if len(unexpected_keys) > 3:
                    print(f"  ... and {len(unexpected_keys) - 3} more")

            net.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully!")

            print("\nAfter loading weights:")
            net.print_model_parameters_info()

        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("This will significantly impact model performance!")
            return
    else:
        print(f"Model file not found: {args.model_path}")
        return

    print("======> Creating test dataloader <======")
    test_image_mask_list = get_image_mask_mapping(config.valid_datasets, flag="valid")
    test_dataset = SegDataset(
        dataset_info_list=test_image_mask_list,
        transforms=data_aug(mode="val"),
        eval_original_resolution=True,
    )
    test_dataloaders = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of batches: {len(test_dataloaders)}")

    print("\n=== Dataset Info ===")
    if hasattr(test_dataset, 'dataset'):
        print(f"Dataset keys: {test_dataset.dataset.keys()}")
        if 'img_path' in test_dataset.dataset:
            print(f"First 3 image paths:")
            for i in range(min(3, len(test_dataset.dataset['img_path']))):
                print(f"  {i}: {test_dataset.dataset['img_path'][i]}")

    print("================> Start testing <================")
    num_results = test_model_with_tta(
        config, net, test_dataloaders, multimask_output,
        args.output_path, args.tta, args.rgb
    )
    print(f"Testing completed! Processed {num_results} samples.")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()