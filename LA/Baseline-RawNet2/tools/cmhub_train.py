#!/usr/bin/env python3
"""CMHub training wrapper for RawNet2."""

import argparse
import json
import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model import RawNet
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train, pad
from main import train_epoch, evaluate_accuracy
from core_scripts.startup_config import set_random_seed
import soundfile as sf


class CMHubDataset(Dataset):
    """Dataset wrapper for CMHub format."""
    
    def __init__(self, scp_path, labels_path):
        self.file_list = []
        self.labels = {}
        
        # Load scp file
        with open(scp_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, wav_path = parts
                else:
                    wav_path = parts[0]
                    utt_id = Path(wav_path).stem
                self.file_list.append((utt_id, wav_path))
        
        # Load labels
        with open(labels_path, 'r') as f:
            label_data = json.load(f)
            for utt_id, label in label_data.items():
                # Convert label to binary: 1 for bonafide, 0 for spoof
                self.labels[utt_id] = 1 if label == 'bonafide' else 0
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        utt_id, wav_path = self.file_list[idx]
        
        # Load audio
        audio, sr = sf.read(wav_path)
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Pad/trim to fixed length
        audio = pad(audio, max_len=64600)
        
        # Get label
        label = self.labels.get(utt_id, 0)
        
        return torch.from_numpy(audio).float(), label


def main():
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument('--train_scp', type=str, required=True)
    parser.add_argument('--dev_scp', type=str, required=True)
    parser.add_argument('--labels', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=1234)
    
    # Wandb parameters
    parser.add_argument('--wandb_enable', type=bool, default=False)
    parser.add_argument('--wandb_run_name', type=str, default='rawnet2_train')
    
    # Backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True)
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_dir = Path(args.out_dir) / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Set random seed
    set_random_seed(args.seed, args)
    
    # Initialize wandb if enabled
    if args.wandb_enable:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'cmhub'),
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Load model config
    config_path = Path(__file__).parent.parent / "model_config_RawNet.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = RawNet(config['model'], device)
    model = model.to(device)
    
    # Create datasets
    train_dataset = CMHubDataset(args.train_scp, args.labels)
    dev_dataset = CMHubDataset(args.dev_scp, args.labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_dev_acc = 0.0
    
    for epoch in range(args.max_epochs):
        print(f'\nEpoch {epoch + 1}/{args.max_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            train_loader, model, args.lr, optimizer, device
        )
        
        # Evaluate
        dev_acc = evaluate_accuracy(dev_loader, model, device)
        
        print(f'\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Dev Acc: {dev_acc:.2f}%')
        
        # Log to wandb
        if args.wandb_enable:
            wandb.log({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'dev_acc': dev_acc,
                'epoch': epoch + 1
            })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'dev_acc': dev_acc,
        }
        
        checkpoint_path = checkpoint_dir / f'epoch_{epoch + 1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_path = checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f'New best model saved with dev accuracy: {dev_acc:.2f}%')
    
    print(f'\nTraining completed. Best dev accuracy: {best_dev_acc:.2f}%')
    
    if args.wandb_enable:
        wandb.finish()


if __name__ == "__main__":
    main()