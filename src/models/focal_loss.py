import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# Adjust path if necessary
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from Enum.PathEnum import PathEnum 
from data.data_helper import load_data 
from Preprocessing.preprocess import Preprocessing 


class FocalLoss(nn.Module):
    """
    Focal Loss for Binary Classification using Logits.
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        prob = pred_logits.sigmoid()
        
        # Calculate alpha_t and p_t based on true class
        alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
        pt = torch.where(target == 1, prob, 1.0 - prob)
        
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class FraudDetectionNN(nn.Module):
    def __init__(self, input_dim: int = 30):
        super(FraudDetectionNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(128, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.Tanh(),

            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_metrics(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            all_preds.append(logits.sigmoid().cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    probs = torch.cat(all_preds, dim=0).numpy()
    return avg_loss, probs


if __name__ == '__main__':
    # Configuration & Setup
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Loading and Scaling
    df_train = load_data(PathEnum.TRAIN_PATH.value)
    df_val = load_data(PathEnum.VAL_PATH.value) 

    preprocessing_pipline = Preprocessing()  

    X_train, y_train, clip_limts, scaler = preprocessing_pipline.fit_transform(df_train)
    X_val, y_val = preprocessing_pipline.transform(df_val, clip_limts, scaler)

    # Convert to Tensor Datasets & DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model Hyperparameters
    alpha = 0.75  # Weighting for positive (fraud) class
    gamma = 2.0   # Focusing parameter for hard samples
    lr = 0.001
    num_epochs = 450

    model = FraudDetectionNN(input_dim=X_train.shape[1]).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Initialize the Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    run_name = f"gamma_{gamma}_alpha_{alpha}_batch_size_{batch_size}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}_SGD_optimizer")

    start_epoch = 0

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs = evaluate_metrics(model, val_loader, criterion, device)

        # Get learning rate before stepping
        current_lr = optimizer.param_groups[0]['lr']
        # Step scheduler
        scheduler.step()

        # TensorBoard Loss & LR Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Log Model Parameters and Gradients
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'{name}.grad', param.grad, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6e}")

        # Checkpointing and Metric Evaluation Every 10 Epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'models/checkpoint_epoch_{epoch + 1}.pth')

            precision, recall, _ = precision_recall_curve(y_val, val_probs)
            auc_pr = auc(recall, precision)

            y_val_pred_default = (val_probs > 0.5).astype(int)
            report_val = classification_report(y_true=y_val, y_pred=y_val_pred_default, output_dict=True, zero_division=0)

            # Safeguard in case "1" (fraud class) is missing in the early epochs
            if "1" in report_val:
                writer.add_scalar('Validation/Precision', report_val["1"]["precision"], epoch + 1)
                writer.add_scalar('Validation/Recall', report_val["1"]["recall"], epoch + 1)
                writer.add_scalar('Validation/F1', report_val["1"]["f1-score"], epoch + 1)
            writer.add_scalar('Validation/AUC_PR', auc_pr, epoch + 1)


    # --- Final Post-Training Evaluation (Properly Indented) ---
    model.eval()
    _, train_probs = evaluate_metrics(model, train_loader, criterion, device)
    _, val_probs = evaluate_metrics(model, val_loader, criterion, device)

    # 1. Default Threshold Evaluation (0.50)
    print("\n--- Model Performance @ Default Threshold (0.50) ---")
    y_val_pred_50 = (val_probs > 0.50).astype(int)
    print("Classification Report:\n", classification_report(y_true=y_val, y_pred=y_val_pred_50, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true=y_val, y_pred=y_val_pred_50))

    # 2. Optimal Threshold Search using sklearn's precision_recall_curve
    print("\n--- Model Performance @ Optimal Threshold ---")
    precisions, recalls, thresholds = precision_recall_curve(y_train, train_probs)

    # Calculate F1-scores across all thresholds and find the max
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_val_pred_opt = (val_probs > optimal_threshold).astype(int)
    print(f"Optimal Threshold Found: {optimal_threshold:.4f}")
    print("Classification Report:\n", classification_report(y_true=y_val, y_pred=y_val_pred_opt, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true=y_val, y_pred=y_val_pred_opt))

    writer.close()