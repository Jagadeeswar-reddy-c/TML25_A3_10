import os, torch, random, requests
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
import time

# --- Constants ---
TOKEN = "49390889"
SUBMIT_URL = "http://34.122.51.94:9090/robustness"
SAVE_PATH = "./models/robust_model.pt"

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img.convert("RGB"))
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class CutoutTransform:
    def __init__(self, size=8, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        # img here is expected to be a Tensor
        if random.random() < self.p:
            h, w = img.size(1), img.size(2)
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)
            img[:, y:y + self.size, x:x + self.size] = 0.0
        return img

def pgd_attack(model, loss_fn, x, y, eps, alpha, steps, random_start=True):
    """
    Optimized PGD attack.
    x and y are assumed to be UNNORMALIZED, in [0, 1] range, and will remain in this range.
    """
    model.eval() # Set model to eval mode during attack generation
    
    # Detach original images and clone for perturbation
    orig_x = x.detach().clone()
    
    if random_start:
        # Initialize with random noise within epsilon ball
        delta = torch.zeros_like(x).uniform_(-eps, eps)
        # Clamp initial delta to keep it within [0,1] bounds from original_x
        perturbed_x = (orig_x + delta).clamp(0, 1)
    else:
        perturbed_x = orig_x.clone()
    
    for _ in range(steps):
        perturbed_x = perturbed_x.clone().requires_grad_(True) # Clone and enable gradients
        output = model(perturbed_x)
        loss = loss_fn(output, y)
        
        grad = torch.autograd.grad(loss, perturbed_x, retain_graph=False, create_graph=False)[0]
        
        # Apply PGD step
        perturbed_x = perturbed_x.detach() + alpha * grad.sign()
        delta = perturbed_x - orig_x # delta is calculated
        delta = torch.clamp(delta, -eps, eps) # Clamp perturbation amount
        # Add delta back to original and clamp to [0, 1] range
        perturbed_x = (orig_x + delta).clamp(0, 1)
    
    model.train() # Set model back to train mode
    return perturbed_x

def fgsm_attack(model, loss_fn, x, y, eps):
    """
    Fast FGSM attack.
    x and y are assumed to be UNNORMALIZED, in [0, 1] range, and will remain in this range.
    """
    model.eval() # Set model to eval mode during attack generation
    
    # Clone the input and enable gradients - don't detach first
    x_adv = x.clone().requires_grad_(True)
    
    # Model now expects [0, 1] input directly
    output = model(x_adv)

    # Compute loss
    loss = loss_fn(output, y)
    
    # Compute gradients
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    
    # Apply FGSM perturbation
    x_adv = x_adv.detach() + eps * grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1) # Clamp to [0,1] range
    
    model.train() # Set model back to train mode
    return x_adv

def build_model(name):
    """Build model with ImageNet pretraining for better initialization"""
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {name}. Must be resnet18, resnet34, or resnet50.")
    
    # Replace final layer for 10 classes (adjust if your dataset has different number of classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # Initialize the new layer
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out')
    nn.init.constant_(model.fc.bias, 0)
    
    # Explicitly ensure all model parameters require gradients
    # This is a safeguard against potential issues where parameters might be frozen.
    for param in model.parameters():
        param.requires_grad = True
        
    return model

def get_dataloader(batch_size=256, is_train=True):
    """
    Enhanced data loaders with separate transforms for training and testing.
    Ensures images are in [0, 1] range directly after ToTensor().
    """
    # Load the dataset
    dataset = torch.load("train.pt", map_location='cpu')
    print("Loaded dataset:", type(dataset))
    print("Number of samples:", len(dataset))
    
    if is_train:
        transform_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(), # Converts to [0,1] range
            CutoutTransform(size=8, p=0.5), # Cutout expects a Tensor
        ])
    else:
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(), # Converts to [0,1] range
        ])
    
    # Apply transform to dataset
    dataset.transform = transform_pipeline
    
    # For simplicity, we'll use the same dataset for both train and test
    # You might want to split this properly based on your needs
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=True)

def mixup_data(x, y, alpha=0.4):
    """Mixup with optimal alpha"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate(model, device, test_loader, attack_type=None, eps=None, alpha=None, steps=None):
    """
    Comprehensive evaluation for clean, FGSM, and PGD.
    Model expects [0, 1] input directly.
    """
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()

    for batch_data in test_loader:
        # Handle the custom dataset format (id, img, label)
        if len(batch_data) == 3:
            _, x, y = batch_data
        else:
            x, y = batch_data
            
        x, y = x.to(device), y.to(device)
        
        # Generate adversarial examples if attack_type is specified
        if attack_type == 'fgsm':
            x_adv = fgsm_attack(model, loss_fn, x, y, eps)
            with torch.no_grad():
                outputs = model(x_adv)
        elif attack_type == 'pgd':
            x_adv = pgd_attack(model, loss_fn, x, y, eps, alpha, steps)
            with torch.no_grad():
                outputs = model(x_adv)
        else: # Clean evaluation
            with torch.no_grad():
                outputs = model(x) # Model expects [0, 1] input directly
            
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    acc = 100 * correct / total
    return acc

def train_epoch_clean(model, loader, optimizer, criterion, device, use_mixup=True):
    """
    Clean training epoch with advanced techniques.
    Model expects [0, 1] input directly.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch_data in enumerate(loader):
        # Handle the custom dataset format (id, img, label)
        if len(batch_data) == 3:
            _, x, y = batch_data
        else:
            x, y = batch_data
            
        x, y = x.to(device), y.to(device)
        
        if use_mixup and random.random() < 0.5:
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
            outputs = model(x_mix) # Model expects [0, 1] input directly
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(x) # Model expects [0, 1] input directly
            loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        if not use_mixup or random.random() >= 0.5: # Simple approx. acc for mixup batches
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    train_acc = 100 * correct / total if total > 0 else 0
    return total_loss / len(loader), train_acc

def train_epoch_adversarial(model, loader, optimizer, criterion, device, config):
    """
    Adversarial training epoch.
    Attacks are applied on and model expects [0, 1] input directly.
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch_data in enumerate(loader):
        # Handle the custom dataset format (id, img, label)
        if len(batch_data) == 3:
            _, x, y = batch_data
        else:
            x, y = batch_data
            
        x, y = x.to(device), y.to(device)
        
        # Clean loss (on [0, 1] clean data)
        clean_output = model(x)
        clean_loss = criterion(clean_output, y)
        
        # Generate adversarial examples (on [0, 1] data)
        # Randomly select between PGD and FGSM as specified by the assignment.
        if random.random() < 0.5: # 50% PGD, 50% FGSM
            x_adv = pgd_attack(model, criterion, x, y, config['eps'], config['alpha'], config['pgd_steps'])
        else:
            x_adv = fgsm_attack(model, criterion, x, y, config['eps'])
        
        # Adversarial loss (on [0, 1] adversarial data)
        adv_output = model(x_adv)
        adv_loss = criterion(adv_output, y)
        
        # Combined loss with balanced weighting
        total_loss_batch = 0.5 * clean_loss + 0.5 * adv_loss
        
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
    
    return total_loss / len(loader)

def save_model(model_state_dict, model_class_name, path):
    """
    Save model's state_dict directly as required by the server.
    The server expects to be able to load this directly into a ResNet model.
    """
    # Ensure the directory exists before saving the file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save only the state_dict directly - the server will handle model creation
    torch.save(model_state_dict, path)
    print(f" Model saved to {path}")

def run_assertions(model_path, model_name):
    """Validate model format as specified in the assignment."""
    # Load the state_dict directly
    loaded_state_dict = torch.load(model_path, map_location="cpu")
    
    # Build model and load state_dict
    model = build_model(model_name)  # Use the model_name parameter
    model.load_state_dict(loaded_state_dict, strict=True)
    
    # Set to evaluation mode for testing
    model.eval()
    
    # Test input dimensionality (model expects [0, 1] input directly)
    test_input = torch.randn(1, 3, 32, 32)
    
    with torch.no_grad():
        out = model(test_input)
    
    # Assert output dimensionality
    assert out.shape == (1, 10), f"Expected output shape (1, 10), got {out.shape}"
    print(" All local assertions passed (model class name, input/output shapes, state_dict loading).")

def submit_model(model_path, model_name):
    """Submit to evaluation server."""
    try:
        with open(model_path, "rb") as f:
            response = requests.post(
                SUBMIT_URL,
                files={"file": f},
                headers={"token": TOKEN, "model-name": model_name}
            )
        print(f" Submission status: {response.status_code}")
        print(f" Server response: {response.json()}")
    except Exception as e:
        print(f" Submission failed: {e}")

def main():
    # Configuration parameters (replaces argparse)
    config = {
        "model_name": "resnet50",
        "epochs": 150,
        "batch_size": 256,
        "lr": 0.1,
        "pgd_steps": 20,
        "eps": 8/255, # Standard epsilon for [0,1] range
        "alpha": 2/255, # Standard alpha for PGD step size
        "no_submit": False, # Set to True to skip submission
        "clean_epochs": 100, # Initial clean training epochs
    }
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    # Model setup
    model = build_model(config['model_name']).to(device)
    print(f" Model: {config['model_name']}")
    
    # Data loaders
    train_loader = get_dataloader(config['batch_size'], is_train=True)
    test_loader = get_dataloader(config['batch_size'], is_train=False)
    
    # Optimizer and criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    # Learning rate scheduler - Apply CosineAnnealingLR for the entire training duration
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0.001)
    
    print(" Starting training...")
    best_clean_acc = 0.0 # Track best clean accuracy for saving model
    best_pgd_acc = 0.0 # Track best PGD accuracy for monitoring
    best_fgsm_acc = 0.0 # Track best FGSM accuracy for monitoring
    
    # Phase 1: Clean training only
    print(f"\n Phase 1: Clean training for {config['clean_epochs']} epochs")
    for epoch in range(1, config['clean_epochs'] + 1):
        loss, train_acc = train_epoch_clean(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        # Evaluate clean accuracy after each epoch or periodically
        test_clean_acc = evaluate(model, device, test_loader, attack_type=None)
        print(f" Epoch {epoch}: Clean Loss={loss:.4f}, Train Acc={train_acc:.2f}%, Test Clean Acc={test_clean_acc:.2f}%")
        
        # Save model if clean accuracy improves AND is above 50% threshold
        if test_clean_acc > best_clean_acc and test_clean_acc >= 50.0:
            best_clean_acc = test_clean_acc
            save_model(model.state_dict(), config['model_name'], SAVE_PATH)
            print(f" New best clean accuracy: {best_clean_acc:.2f}%. Model saved.")
    
    # Phase 2: Adversarial training
    print(f"\n Phase 2: Adversarial training for {config['epochs'] - config['clean_epochs']} epochs")
    for epoch in range(config['clean_epochs'] + 1, config['epochs'] + 1):
        loss = train_epoch_adversarial(model, train_loader, optimizer, criterion, device, config)
        scheduler.step()
        
        # Evaluate all accuracies periodically
        test_clean_acc = evaluate(model, device, test_loader, attack_type=None)
        test_fgsm_acc = evaluate(model, device, test_loader, attack_type='fgsm', eps=config['eps'])
        test_pgd_acc = evaluate(model, device, test_loader, attack_type='pgd', eps=config['eps'], alpha=config['alpha'], steps=config['pgd_steps'])
        
        print(f" Epoch {epoch}: Adv Loss={loss:.4f}, Test Clean Acc={test_clean_acc:.2f}%, FGSM Rob={test_fgsm_acc:.2f}%, PGD Rob={test_pgd_acc:.2f}%")
        
        # Save model if clean accuracy improves AND is above 50% threshold,
        # OR if robustness greatly improves while clean accuracy is still acceptable.
        if test_clean_acc >= 50.0 and (test_clean_acc > best_clean_acc or \
           (test_clean_acc >= best_clean_acc * 0.95 and (test_pgd_acc > best_pgd_acc or test_fgsm_acc > best_fgsm_acc))):
            best_clean_acc = test_clean_acc
            best_pgd_acc = test_pgd_acc
            best_fgsm_acc = test_fgsm_acc
            save_model(model.state_dict(), config['model_name'], SAVE_PATH)
            print(f" New best performance found (Clean: {test_clean_acc:.2f}%, FGSM: {test_fgsm_acc:.2f}%, PGD: {test_pgd_acc:.2f}%). Model saved.")
        elif test_clean_acc > best_clean_acc and test_clean_acc >= 50.0:
            best_clean_acc = test_clean_acc
            save_model(model.state_dict(), config['model_name'], SAVE_PATH)
            print(f" New best clean accuracy: {best_clean_acc:.2f}%. Model saved.")

    print("\n Final evaluation of the best saved model...")
    try:
        # Load the state_dict directly
        loaded_state_dict = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(loaded_state_dict)
        model.to(device)
        
        final_clean_acc = evaluate(model, device, test_loader, attack_type=None)
        final_fgsm_acc = evaluate(model, device, test_loader, attack_type='fgsm', eps=config['eps'])
        final_pgd_acc = evaluate(model, device, test_loader, attack_type='pgd', eps=config['eps'], alpha=config['alpha'], steps=config['pgd_steps'])
        
        print(f"Final Clean Accuracy: {final_clean_acc:.2f}%")
        print(f"Final FGSM Robustness: {final_fgsm_acc:.2f}%")
        print(f"Final PGD Robustness: {final_pgd_acc:.2f}%")
        
        run_assertions(SAVE_PATH, config['model_name'])
        
        if not config['no_submit']:
            submit_model(SAVE_PATH, config['model_name'])
        
    except FileNotFoundError:
        print(f" Error: Model file not found at {SAVE_PATH}. Ensure training completed and model was saved.")
    except Exception as e:
        print(f" An error occurred during final evaluation or submission: {e}")

    print("\n Training and evaluation process completed!")

if __name__ == "__main__":
    main()