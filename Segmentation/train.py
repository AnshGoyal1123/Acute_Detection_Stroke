import os
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from Models import AUIS, Classic_AttUNet, UNet3D, ResUNet3D, SwinUNETR, DynUNet
from Dataset.Dataset import StrokeAI
from monai.losses import DiceLoss
import torch.nn.functional as F
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

def get_args():
    parser = argparse.ArgumentParser(description="Train Stroke Segmentation Models (DDP)")
    parser.add_argument('--model_name', type=str, required=True, choices=['AUIS', 'Classic_AttUNet', 'UNet3D', 'ResUNet3D', 'SwinUNETR', 'DynUNet'])
    parser.add_argument('--ct_root', type=str, default="/home/agoyal19/Datasets/Segmentation_Dataset/images")
    parser.add_argument('--label_root', type=str, default="/home/agoyal19/Datasets/Segmentation_Dataset/labels")
    parser.add_argument('--adc_root', type=str, default="/scratch4/rsteven1/ADC_coregis_20231228")
    parser.add_argument('--dwi_root', type=str, default="/scratch4/rsteven1/StrokeAI/DWI_coregis")
    parser.add_argument('--mapping_file', type=str, default="/home/agoyal19/My_Work/Segmentation/mapping.json")
    parser.add_argument('--mri_type', type=str, default='ADC', choices=['ADC', 'DWI', 'None'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--rot_aug', action='store_true')
    parser.add_argument('--instance_norm', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    return parser.parse_args()

def get_model(name):
    model_dict = {
        'AUIS': AUIS.AUIS,
        'Classic_AttUNet': Classic_AttUNet.AttUnet,
        'UNet3D': UNet3D.UNet3D,
        'ResUNet3D': ResUNet3D.ResUNet3D,
        'SwinUNETR': SwinUNETR.SwinUNETR,
        'DynUNet': DynUNet.DynUNet
    }
    return model_dict[name]()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_dataloaders(args, rank, world_size):
    train_dataset = StrokeAI(
        CT_root=args.ct_root,
        DWI_root=args.dwi_root,
        ADC_root=args.adc_root,
        label_root=args.label_root,
        MRI_type=args.mri_type,
        mode='train',
        map_file=args.mapping_file,
        crop=args.crop,
        instance_normalize=args.instance_norm,
        RotatingResize=args.rot_aug,
    )
    test_dataset = StrokeAI(
        CT_root=args.ct_root,
        DWI_root=args.dwi_root,
        ADC_root=args.adc_root,
        label_root=args.label_root,
        MRI_type=args.mri_type,
        mode='test',
        map_file=args.mapping_file,
        crop=args.crop,
        instance_normalize=args.instance_norm,
        RotatingResize=False
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=2)
    return train_loader, test_loader

def train_model(rank, world_size, args):
    setup(rank, world_size)

    model = get_model(args.model_name).to(rank)
    model = DDP(model, device_ids=[rank])

    train_loader, test_loader = get_dataloaders(args, rank, world_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceLoss(sigmoid=False)

    if rank == 0 and args.wandb and wandb:
        wandb.init(project="StrokeAI", name=args.model_name)
        wandb.config.update(args)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            ct = batch['ct'].to(rank)
            label = batch['label'].to(rank)

            pred = model(ct)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * ct.size(0)

        avg_loss = torch.tensor(total_loss / len(train_loader.dataset), device=rank)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / world_size

        model.eval()
        dice_scores = []
        with torch.no_grad():
            for batch in test_loader:
                ct = batch['ct'].to(rank)
                label = batch['label'].to(rank)

                pred = model(ct)
                pred_binary = (pred > 0.5).float()

                intersection = (pred_binary * label).sum(dim=(1,2,3,4))
                union = pred_binary.sum(dim=(1,2,3,4)) + label.sum(dim=(1,2,3,4))
                dice = (2. * intersection + 1e-6) / (union + 1e-6)

                dice_scores.extend(dice.cpu().numpy().tolist())

        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)

        if rank == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss.item():.4f}, Dice = {mean_dice:.4f} ± {std_dice:.4f}")
            if args.wandb and wandb:
                wandb.log({"epoch": epoch+1, "train_loss": avg_loss.item(), "val_dice": mean_dice, "val_dice_std": std_dice})

    cleanup()

def main():
    args = get_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train_model, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()