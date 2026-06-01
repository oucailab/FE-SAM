import os
import random
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch.backends.cudnn as cudnn
from importlib import import_module

from utils.py2cfg import py2cfg
from utils.metric import Evaluator
from utils.loss import dice_ce_loss
from utils.dataloader import SegDataset
from utils.dataloader import get_image_mask_mapping
from utils.transforms import data_augmentation as data_aug
from models.segment_anything import sam_model_registry

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def write_to_log(log_file, data):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(f"[{current_time}] Log file created.\n")

    print(f"[{current_time}] {data}")
    with open(log_file, 'a') as f:
        if data.startswith("Start"):
            f.write(f"\n")
        f.write(f"[{current_time}] {data}\n")


def train(config, model, train_dataloaders, valid_dataloaders, multimask_output):
    base_lr = config.learning_rate
    learning_rate = base_lr / config.warmup_period if config.warmup else base_lr

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop_epoch)
    lr_scheduler.last_epoch = config.start_epoch

    writer = SummaryWriter(config.tensorboard_path)
    max_iterations = config.max_epoch_num * len(train_dataloaders)
    best_model_epoch, best_miou, best_f1, best_oa = 0, 0, 0, 0
    iter_num = 0

    for epoch in range(config.start_epoch, config.max_epoch_num):
        model.train()
        print("\n" + "+" * 36 + " Epoch: " + str(epoch).zfill(3) + " " + "+" * 36)
        model.print_model_parameters_info()
        data = f"Start Training Epoch: {str(epoch).zfill(3)}, Learning Rate: {optimizer.param_groups[0]['lr']}"
        write_to_log(f"{config.output_path}/train_logs.txt", data)
        iterator = tqdm(train_dataloaders)

        for batch in iterator:
            images, labels = batch['image'], batch['label']
            if torch.cuda.is_available() and config.device != "cpu":
                images, labels = images.cuda(), labels.cuda()

            outputs, intermed_embeddings = model(images, multimask_output, config.image_size[0])
            loss, loss_ce, loss_dice = dice_ce_loss(outputs["masks"], labels, config.num_classes, config.dice_param)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config.warmup and iter_num < config.warmup_period:
                lr = base_lr * ((iter_num + 1) / config.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                if config.warmup:
                    shift_iter = iter_num - config.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            iterator.set_postfix(loss=loss.item(), loss_ce=loss_ce.item(), loss_dice=loss_dice.item())
            iter_num += 1

            writer.add_scalar('info/lr', lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

        if epoch >= config.stop_epoch:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            model_name = config.checkpoint_name.replace("epoch", str(epoch).zfill(3))
            if not os.path.exists(config.output_path):
                os.makedirs(config.output_path)
            model_path = f"{config.output_path}/{model_name}"

            original_batch_size = config.valid_batch_size
            config.valid_batch_size = 1

            try:
                f1, miou, oa, coarse_f1, coarse_miou, coarse_oa = inference(config, model, valid_dataloaders, multimask_output)
            except RuntimeError as e:
                print(f"Validation failed with error: {e}")
                f1, miou, oa, coarse_f1, coarse_miou, coarse_oa = 0, 0, 0, 0, 0, 0
            finally:
                config.valid_batch_size = original_batch_size

            writer.add_scalar('info/f1', f1, epoch)
            writer.add_scalar('info/miou', miou, epoch)
            writer.add_scalar('info/oa', oa, epoch)
            writer.add_scalar('info/coarse_f1', coarse_f1, epoch)
            writer.add_scalar('info/coarse_miou', coarse_miou, epoch)
            writer.add_scalar('info/coarse_oa', coarse_oa, epoch)

            if epoch % config.model_save_fre == 0:
                try:
                    torch.save(model.state_dict(), model_path)
                    write_to_log(f"{config.output_path}/train_logs.txt", f"Model saved to {model_path}")
                except Exception as e:
                    write_to_log(f"{config.output_path}/train_logs.txt", f"Failed to save model: {e}")

            if miou > best_miou:
                best_model_epoch, best_miou, best_f1, best_oa = epoch, miou, f1, oa
                model_name = config.checkpoint_name.replace("epoch", "best")
                model_path = f"{config.output_path}/{model_name}"

                try:
                    torch.save(model.state_dict(), model_path)
                    data = f"new best model epoch - {best_model_epoch}: F1:{f1}, mIOU:{miou}, OA:{oa}"
                    write_to_log(f"{config.output_path}/train_logs.txt", data)
                    write_to_log(f"{config.output_path}/train_logs.txt", f"Best model saved to {model_path}")
                except Exception as e:
                    write_to_log(f"{config.output_path}/train_logs.txt", f"Failed to save best model: {e}")

        lr_scheduler.step()

    writer.close()


def inference(config, model, valid_dataloaders, multimask_output):
    model.eval()

    evaluator_enhanced = Evaluator(num_class=config.num_classes)
    evaluator_coarse = Evaluator(num_class=config.num_classes)
    evaluator_enhanced.reset()
    evaluator_coarse.reset()

    iterator = tqdm(valid_dataloaders, desc="Validation")

    with torch.no_grad():
        for batch in iterator:
            idx, images, labels = batch['idx'], batch['image'], batch['label']

            for i in range(images.size(0)):
                single_image = images[i:i+1]
                single_label = labels[i:i+1]

                if torch.cuda.is_available() and config.device != "cpu":
                    single_image = single_image.cuda()

                try:
                    predictions, _ = model(single_image, multimask_output, config.image_size[0])

                    enhanced_predictions = torch.nn.Softmax(dim=1)(predictions["masks"])
                    enhanced_mask = enhanced_predictions.argmax(dim=1)[0].cpu().numpy()

                    if "coarse_masks" in predictions:
                        coarse_predictions = torch.nn.Softmax(dim=1)(predictions["coarse_masks"])
                        coarse_mask = coarse_predictions.argmax(dim=1)[0].cpu().numpy()
                        has_coarse = True
                    else:
                        print("Warning: No coarse_masks found, using enhanced_masks")
                        coarse_mask = enhanced_mask
                        has_coarse = False

                    label = single_label[0].cpu().numpy()
                    evaluator_enhanced.add_batch(pre_image=enhanced_mask, gt_image=label)
                    evaluator_coarse.add_batch(pre_image=coarse_mask, gt_image=label)

                    del predictions, single_image, enhanced_predictions
                    if has_coarse:
                        del coarse_predictions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    print(f"Skipping validation sample due to memory error: {e}")
                    continue

    iou_per_class_enhanced = evaluator_enhanced.Intersection_over_Union()
    f1_per_class_enhanced = evaluator_enhanced.F1()
    oa_enhanced = evaluator_enhanced.OA()

    iou_per_class_coarse = evaluator_coarse.Intersection_over_Union()
    f1_per_class_coarse = evaluator_coarse.F1()
    oa_coarse = evaluator_coarse.OA()

    write_to_log(f"{config.output_path}/train_logs.txt", "\n=== ENHANCED MASKS RESULTS ===")
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class_enhanced, f1_per_class_enhanced):
        data = f'Enhanced_F1_{class_name}:{class_f1:.4f}, Enhanced_IOU_{class_name}:{class_iou:.4f}'
        write_to_log(f"{config.output_path}/train_logs.txt", data)

    enhanced_f1_mean = np.nanmean(f1_per_class_enhanced[:-1])
    enhanced_miou_mean = np.nanmean(iou_per_class_enhanced[:-1])

    write_to_log(f"{config.output_path}/train_logs.txt", "\n=== COARSE MASKS RESULTS ===")
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class_coarse, f1_per_class_coarse):
        data = f'Coarse_F1_{class_name}:{class_f1:.4f}, Coarse_IOU_{class_name}:{class_iou:.4f}'
        write_to_log(f"{config.output_path}/train_logs.txt", data)

    coarse_f1_mean = np.nanmean(f1_per_class_coarse[:-1])
    coarse_miou_mean = np.nanmean(iou_per_class_coarse[:-1])

    f1_improvement = enhanced_f1_mean - coarse_f1_mean
    miou_improvement = enhanced_miou_mean - coarse_miou_mean
    oa_improvement = oa_enhanced - oa_coarse

    write_to_log(f"{config.output_path}/train_logs.txt", "\n=== PERFORMANCE COMPARISON ===")
    write_to_log(f"{config.output_path}/train_logs.txt", f'Enhanced - F1:{enhanced_f1_mean:.4f}, mIOU:{enhanced_miou_mean:.4f}, OA:{oa_enhanced:.4f}')
    write_to_log(f"{config.output_path}/train_logs.txt", f'Coarse   - F1:{coarse_f1_mean:.4f}, mIOU:{coarse_miou_mean:.4f}, OA:{oa_coarse:.4f}')
    write_to_log(f"{config.output_path}/train_logs.txt", f'Improvement - F1:{f1_improvement:+.4f}, mIOU:{miou_improvement:+.4f}, OA:{oa_improvement:+.4f}')

    return enhanced_f1_mean, enhanced_miou_mean, oa_enhanced, coarse_f1_mean, coarse_miou_mean, oa_coarse


if __name__ == "__main__":
    os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

    config = py2cfg("/root/autodl-tmp/SAM-Adapter-main/configs/adapter_sam_pd.py")
    seed_everything(config.seed)
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

    print("======> Creating training dataloader <======")
    train_image_mask_list = get_image_mask_mapping(config.train_datasets, flag="train")
    train_dataset = SegDataset(
        dataset_info_list=train_image_mask_list,
        transforms=data_aug(mode="train"),
        eval_original_resolution=False,
    )
    train_dataloaders = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        drop_last=True,
        num_workers=0,
    )

    valid_image_mask_list = get_image_mask_mapping(config.valid_datasets, flag="valid")
    valid_dataset = SegDataset(
        dataset_info_list=valid_image_mask_list,
        transforms=data_aug(mode="val"),
        eval_original_resolution=True,
    )
    valid_dataloaders = DataLoader(
        dataset=valid_dataset,
        batch_size=config.valid_batch_size,
        drop_last=True,
        num_workers=0,
    )

    print("================> Start training <================")
    train(config, net, train_dataloaders, valid_dataloaders, multimask_output)
    print("Training is done!")