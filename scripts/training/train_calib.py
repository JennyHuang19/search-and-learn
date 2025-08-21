import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ----- 1. Dataset Definition -----
class SoftLabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- 2. MLP Model -----
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.GELU(),
            nn.Linear(200, 200),
            nn.GELU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.model(x)

# ----- 3. Platt Calibrator -----
class PlattCalibrator(nn.Module):
    """p = sigmoid(a * logit + b)"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, logits):
        return self.a * logits + self.b
    
    def predict_proba(self, logits):
        return torch.sigmoid(self.forward(logits))

@torch.no_grad()
def collect_logits_and_labels(model, loader, device="cuda"):
    model.eval()
    all_logits, all_y = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        all_logits.append(logits.squeeze(1).cpu())
        all_y.append(y.squeeze(1).cpu())
    return torch.cat(all_logits), torch.cat(all_y)

@torch.no_grad()
def predict_calibrated_proba(model, calibrator, X, device="cuda"):
    model.eval()
    X = X.to(device)
    logits = model(X)
    return calibrator.predict_proba(logits).squeeze(1)

def train_and_calibrate(X_train, y_train, X_test, y_test, output_dir="outputs"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ----- 4. Dataloaders -----
    train_dataset = SoftLabelDataset(X_train, y_train)
    val_dataset = SoftLabelDataset(X_test, y_test)
    
    # Calibration split
    calib_frac = 0.2
    calib_size = int(len(train_dataset) * calib_frac)
    train_size = len(train_dataset) - calib_size
    
    train_dataset, calib_dataset = random_split(train_dataset, [train_size, calib_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=False)
    
    # ----- 5. Model, Loss, and Optimizer -----
    model = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # ----- 6. Training Loop -----
    num_epochs = 10
    patience = 1
    
    agg_train_loss = []
    agg_val_loss = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print("Training model...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if (len(train_loss) % 100) == 0:
                print(f"Train loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_preds = model(val_X)
                val_loss.append(criterion(val_preds, val_y).item())
        
        mean_train_loss = np.mean(train_loss)
        mean_val_loss = np.mean(val_loss)
        
        agg_train_loss.append(mean_train_loss)
        agg_val_loss.append(mean_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {mean_train_loss:.4f}, Val Loss: {mean_val_loss:.4f}")
        
        # Early stopping
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # ----- 7. Plot Training and Validation Loss -----
    plt.figure(figsize=(10, 5))
    plt.plot(agg_train_loss, label='Train Loss', color='blue')
    plt.plot(agg_val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----- 8. Calibration -----
    print("Calibrating model...")
    logits_cal, y_cal = collect_logits_and_labels(model, calib_loader, device)
    
    calibrator = PlattCalibrator().to(device)
    bce_logits = nn.BCEWithLogitsLoss()
    opt = optim.LBFGS(calibrator.parameters(), lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    
    def closure():
        opt.zero_grad()
        z = calibrator(logits_cal.to(device))
        loss = bce_logits(z, y_cal.to(device))
        loss.backward()
        return loss
    
    opt.step(closure)
    
    print(f"Platt params: a={calibrator.a.item():.4f}, b={calibrator.b.item():.4f}")
    
    # ----- 9. Generate Calibrated Probabilities -----
    print("Generating calibrated probabilities...")
    probs = []
    ys = []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            p = predict_calibrated_proba(model, calibrator, X, device)
            probs.append(p.cpu())
            ys.append(y.squeeze(1).cpu())
    
    calibrated_probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()
    
    # ----- 10. Plot Predicted vs True -----
    plt.figure(figsize=(8, 6))
    plt.scatter(ys, calibrated_probs, alpha=0.005)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Probabilities')
    plt.title('Predicted vs True (Soft Label)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'predicted_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ----- 11. Save Calibrated Probabilities -----
    np.save(os.path.join(output_dir, 'calibrated_probs.npy'), calibrated_probs)
    np.save(os.path.join(output_dir, 'true_labels.npy'), ys)
    
    print(f"Results saved to {output_dir}/")
    print(f"Calibrated probabilities shape: {calibrated_probs.shape}")
    
    return calibrated_probs

def main():
    parser = argparse.ArgumentParser(description='Train and calibrate MLP model')
    parser.add_argument('--X_train', type=str, required=True, help='Path to X_train.npy')
    parser.add_argument('--y_train', type=str, required=True, help='Path to y_train.npy')
    parser.add_argument('--X_test', type=str, required=True, help='Path to X_test.npy')
    parser.add_argument('--y_test', type=str, required=True, help='Path to y_test.npy')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    X_train = np.load(args.X_train)
    y_train = np.load(args.y_train)
    X_test = np.load(args.X_test)
    y_test = np.load(args.y_test)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Train and calibrate
    calibrated_probs = train_and_calibrate(X_train, y_train, X_test, y_test, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
