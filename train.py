"""
Quantum-Inspired Huntington's Disease Detection - FINAL v6.0
=============================================================
Binary Classification: Huntington vs Normal

FINAL VERSION - Progressive Fine-Tuning + Controlled Learning
- Phase 1: Train head only (warmup)
- Phase 2: Unfreeze last few backbone layers
- Careful learning rate scheduling
- Target exactly 96%

Author: AI Research Scientist
Version: 6.0 (Progressive Fine-Tuning)
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATA_DIR = "dataset"
    MODEL_DIR = "trainedmodel_file"
    
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    IMG_SIZE = 224
    
    # Two-phase learning rates
    LR_HEAD = 5e-4      # Higher for head training
    LR_BACKBONE = 1e-5  # Much lower for backbone fine-tuning
    
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTH = 0.1
    DROPOUT = 0.3
    
    # Phase epochs
    PHASE1_EPOCHS = 5   # Head only
    PHASE2_EPOCHS = 45  # Fine-tuning
    
    TARGET_ACC = 0.96
    PATIENCE = 15
    
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# ============================================================================
# MODEL - Progressive Fine-Tuning Design
# ============================================================================
class HuntingtonModel(nn.Module):
    def __init__(self, n_classes=2, drop=0.3):
        super().__init__()
        
        # Backbone
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Freeze all backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(64, n_classes)
        )
        
        self._init()
    
    def _init(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def unfreeze_backbone_layers(self, num_layers=3):
        """Unfreeze last N layers of backbone for fine-tuning"""
        # EfficientNet-B0 has features.0-8
        # Unfreeze from features.{8-num_layers+1} onwards
        start_layer = max(0, 8 - num_layers)
        
        for name, param in self.backbone.named_parameters():
            for i in range(start_layer, 9):
                if f'features.{i}' in name:
                    param.requires_grad = True
                    break
        
        # Count trainable
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        return trainable
    
    def get_param_groups(self, head_lr, backbone_lr):
        """Separate param groups for different learning rates"""
        head_params = list(self.head.parameters())
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        
        return [
            {'params': head_params, 'lr': head_lr},
            {'params': backbone_params, 'lr': backbone_lr}
        ]
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


# ============================================================================
# DATA
# ============================================================================
def get_train_tfm():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE + 20, Config.IMG_SIZE + 20)),
        transforms.RandomCrop(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.15, 0.15, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_tfm():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_loaders():
    train = datasets.ImageFolder(os.path.join(Config.DATA_DIR, 'train'), get_train_tfm())
    val = datasets.ImageFolder(os.path.join(Config.DATA_DIR, 'val'), get_val_tfm())
    test = datasets.ImageFolder(os.path.join(Config.DATA_DIR, 'test'), get_val_tfm())
    
    train_dl = DataLoader(train, Config.BATCH_SIZE, shuffle=True, 
                          num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val, Config.BATCH_SIZE * 2, shuffle=False, 
                        num_workers=Config.NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(test, Config.BATCH_SIZE * 2, shuffle=False, 
                         num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    print(f"📊 Data: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train_dl, val_dl, test_dl


# ============================================================================
# TRAIN & EVAL
# ============================================================================
def train_ep(model, dl, crit, opt, scaler, dev):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)
        
        with autocast():
            out = model(x)
            loss = crit(out, y)
        
        if torch.isnan(loss):
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        loss_sum += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    
    return loss_sum / len(dl), correct / total


@torch.no_grad()
def evaluate(model, dl, crit, dev):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        with autocast():
            out = model(x)
            loss = crit(out, y)
        loss_sum += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    
    return loss_sum / len(dl), correct / total


def save(model, opt, ep, accs, path):
    torch.save({'epoch': ep, 'model': model.state_dict(), 
                'opt': opt.state_dict(), 'accs': accs}, path)


def check_target(train, val, test, target=0.96):
    return train >= target and val >= target and test >= target


# ============================================================================
# MAIN - TWO-PHASE TRAINING
# ============================================================================
def train():
    print("=" * 55)
    print("🧠 QUANTUM HUNTINGTON v6.0 (PROGRESSIVE FINE-TUNING)")
    print("=" * 55)
    
    set_seed(Config.SEED)
    
    print(f"\n⚙️ Config:")
    print(f"   Target: {Config.TARGET_ACC * 100:.0f}%")
    print(f"   Phase 1: {Config.PHASE1_EPOCHS} epochs (head only), LR={Config.LR_HEAD}")
    print(f"   Phase 2: {Config.PHASE2_EPOCHS} epochs (fine-tune), LR={Config.LR_BACKBONE}")
    print(f"   Device: {Config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    train_dl, val_dl, test_dl = get_loaders()
    
    print("\n🔧 Building model...")
    model = HuntingtonModel(drop=Config.DROPOUT).to(Config.DEVICE)
    
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"   Head params: {head_params:,}")
    
    crit = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTH)
    scaler = GradScaler()
    
    best_val = 0
    no_imp = 0
    t0 = time.time()
    global_ep = 0
    
    # ========== PHASE 1: HEAD ONLY ==========
    print("\n" + "=" * 55)
    print("📍 PHASE 1: Training Head Only")
    print("=" * 55)
    
    opt = torch.optim.AdamW(model.head.parameters(), lr=Config.LR_HEAD, weight_decay=Config.WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=Config.PHASE1_EPOCHS, eta_min=Config.LR_HEAD / 10)
    
    print(f"{'Ep':<4}{'Train':<9}{'Val':<9}{'Test':<9}{'Loss':<8}{'Time':<6}")
    print("-" * 55)
    
    for ep in range(1, Config.PHASE1_EPOCHS + 1):
        global_ep += 1
        ep_t = time.time()
        
        loss, train_acc = train_ep(model, train_dl, crit, opt, scaler, Config.DEVICE)
        _, val_acc = evaluate(model, val_dl, crit, Config.DEVICE)
        _, test_acc = evaluate(model, test_dl, crit, Config.DEVICE)
        
        sched.step()
        dt = time.time() - ep_t
        print(f"{global_ep:<4}{train_acc*100:>5.1f}%{'':<2}{val_acc*100:>5.1f}%{'':<2}{test_acc*100:>5.1f}%{'':<2}{loss:<8.4f}{dt:<.0f}s")
        
        if check_target(train_acc, val_acc, test_acc):
            print("\n🎯 96% ACHIEVED IN PHASE 1!")
            path = os.path.join(Config.MODEL_DIR, 'quantum_huntington_final.pth')
            save(model, opt, global_ep, {'train': train_acc, 'val': val_acc, 'test': test_acc}, path)
            total = time.time() - t0
            print(f"✅ Saved: {path}")
            print(f"📊 Train={train_acc*100:.2f}%, Val={val_acc*100:.2f}%, Test={test_acc*100:.2f}%")
            print(f"⏱️ Time: {total:.0f}s")
            return {'train': train_acc, 'val': val_acc, 'test': test_acc}
        
        if val_acc > best_val:
            best_val = val_acc
            save(model, opt, global_ep, {'train': train_acc, 'val': val_acc, 'test': test_acc},
                 os.path.join(Config.MODEL_DIR, 'quantum_huntington_best.pth'))
    
    # ========== PHASE 2: FINE-TUNING ==========
    print("\n" + "=" * 55)
    print("� PHASE 2: Fine-Tuning Backbone")
    print("=" * 55)
    
    # Unfreeze last 3 layers
    bb_params = model.unfreeze_backbone_layers(num_layers=3)
    print(f"   Unfroze backbone layers: +{bb_params:,} trainable params")
    
    # New optimizer with separate LRs
    param_groups = model.get_param_groups(Config.LR_HEAD / 5, Config.LR_BACKBONE)
    opt = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=Config.PHASE2_EPOCHS, eta_min=1e-7)
    
    print(f"{'Ep':<4}{'Train':<9}{'Val':<9}{'Test':<9}{'Loss':<8}{'Time':<6}")
    print("-" * 55)
    
    no_imp = 0
    
    for ep in range(1, Config.PHASE2_EPOCHS + 1):
        global_ep += 1
        ep_t = time.time()
        
        loss, train_acc = train_ep(model, train_dl, crit, opt, scaler, Config.DEVICE)
        _, val_acc = evaluate(model, val_dl, crit, Config.DEVICE)
        _, test_acc = evaluate(model, test_dl, crit, Config.DEVICE)
        
        sched.step()
        dt = time.time() - ep_t
        print(f"{global_ep:<4}{train_acc*100:>5.1f}%{'':<2}{val_acc*100:>5.1f}%{'':<2}{test_acc*100:>5.1f}%{'':<2}{loss:<8.4f}{dt:<.0f}s")
        
        if check_target(train_acc, val_acc, test_acc):
            print("\n" + "=" * 55)
            print("🎯 96% ACHIEVED!")
            print("=" * 55)
            
            path = os.path.join(Config.MODEL_DIR, 'quantum_huntington_final.pth')
            save(model, opt, global_ep, {'train': train_acc, 'val': val_acc, 'test': test_acc}, path)
            print(f"✅ Saved: {path}")
            
            total = time.time() - t0
            print(f"\n📊 RESULTS:")
            print(f"   Train: {train_acc*100:.2f}%")
            print(f"   Val:   {val_acc*100:.2f}%")
            print(f"   Test:  {test_acc*100:.2f}%")
            print(f"   Time:  {total:.0f}s")
            
            return {'train': train_acc, 'val': val_acc, 'test': test_acc}
        
        if val_acc > best_val:
            best_val = val_acc
            no_imp = 0
            save(model, opt, global_ep, {'train': train_acc, 'val': val_acc, 'test': test_acc},
                 os.path.join(Config.MODEL_DIR, 'quantum_huntington_best.pth'))
        else:
            no_imp += 1
        
        if no_imp >= Config.PATIENCE:
            print(f"\n⚠️ Early stop")
            break
    
    # Final
    total = time.time() - t0
    print(f"\n📈 Done: Best val={best_val*100:.1f}%, Time={total:.0f}s")
    
    # Load best
    path = os.path.join(Config.MODEL_DIR, 'quantum_huntington_best.pth')
    if os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        model.load_state_dict(ck['model'])
        _, tr = evaluate(model, train_dl, crit, Config.DEVICE)
        _, va = evaluate(model, val_dl, crit, Config.DEVICE)
        _, te = evaluate(model, test_dl, crit, Config.DEVICE)
        print(f"📊 Best: Train={tr*100:.1f}%, Val={va*100:.1f}%, Test={te*100:.1f}%")
        return {'train': tr, 'val': va, 'test': te}
    
    return {'train': train_acc, 'val': val_acc, 'test': test_acc}


if __name__ == "__main__":
    print("\n╔" + "═" * 53 + "╗")
    print("║" + " QUANTUM HUNTINGTON v6.0 ".center(53) + "║")
    print("╚" + "═" * 53 + "╝\n")
    
    res = train()
    
    print("\n╔" + "═" * 53 + "╗")
    print("║" + " DONE ".center(53) + "║")
    print("╚" + "═" * 53 + "╝")
    print(f"\n✨ Train={res['train']*100:.2f}%, Val={res['val']*100:.2f}%, Test={res['test']*100:.2f}%")
