import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  


from Enum.PathEnum import PathEnum 
from data.data_helper import load_data
from Preprocessing.preprocess import Preprocessing 

from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns 
from numpy import argmax


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        # focal loss : https://leimao.github.io/blog/Focal-Loss-Explained/
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_logits, target):
        BCELoss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        prob = pred_logits.sigmoid()
        alpha_t = torch.where(target == 1, self.alpha, (1 - self.alpha))
        pt =  torch.where(target == 1, prob, 1 - prob)
        loss = alpha_t * ((1 - pt) ** self.gamma) * BCELoss
        return loss.sum()

class FraudDetectionNN(nn.Module):
    def __init__(self):
        super(FraudDetectionNN, self).__init__()
        self.hidden1 = nn.Linear(30, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.hidden2 = nn.Linear(128, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.hidden3 = nn.Linear(128, 16, bias=False)
        self.bn3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 1)
        self.tanh = nn.Tanh()
        self.dorpout = nn.Dropout(0.5)
    

    def forward(self, x):
        x = self.tanh(self.bn1(self.hidden1(x)))
        x = self.dorpout(x)
        x = self.tanh(self.bn2(self.hidden2(x)))
        x = self.dorpout(x)
        x = self.tanh(self.bn3(self.hidden3(x)))
        x = self.output(x)
        return x

# Helper functions 
def eval_classification_report_confusion_matrix(y_pred, y_true, title="" ,save_png=False, path="", digits=5 ):

    print(f'{title} Classification Report')
    print(classification_report(y_pred=y_pred, y_true=y_true, digits=digits))   
    report_stats = classification_report(y_pred=y_pred, y_true=y_true, digits=digits, output_dict=True)

    labels = ['True Negative', 'False Positive' , 'False Negative', 'True Positive'] # order of confusion matrix labels
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cm_flat = cm.flatten()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    for i, txt in enumerate(cm_flat):
        plt.text(i % 2 + 0.5, i // 2 + 0.5, f"{labels[i]}\n{txt}", ha='center', va='center', color='black')
    plt.title(f'Confusion Matrix of {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    if save_png: 
        plt.savefig(f'{path}/{title} Confusion Matrix.png')
    else: 
        plt.show()

    return report_stats


def eval_best_threshold(y_pred,y_true , with_repect_to="f1_score"): 
    """
    Get best threshold from precision recall curve with respect to f1_score, precision or recall

    parameters:
    y_pred: predicted values
    y_true: true values
    with_repect_to: "f1_score" , "precision" or "recall"

    returns:
    optimal threshold and f1 scores
    """
    precision, recall, thresholds = precision_recall_curve(y_score=y_pred,y_true=y_true)
    f1_scores = ((2 * precision * recall) / (precision + recall))

    if with_repect_to == "f1_score":
        optimal_threshold_index = argmax(f1_scores)
    elif with_repect_to == "precision":
        optimal_threshold_index = argmax(precision)
    elif with_repect_to == "recall":
        optimal_threshold_index = argmax(recall)
    else:
        raise ValueError("Invalid value for with_repect_to. Please choose 'f1_score', 'precision' or 'recall'.")        

    optimal_threshold = thresholds[optimal_threshold_index]
    print("Optimal Threshold:", optimal_threshold , "F1 Score:", f1_scores[optimal_threshold_index])
    return optimal_threshold , f1_scores

def save_checkpoint(model, epoch, checkpoint_dir='models/focal_loss_checkpoints', title=''):
    checkpoint_dir = checkpoint_dir + title
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


def eval_auc_precision_recall_curve(y_pred_prob, y_true):
     """
     Get Area under curve of precision recal of precision recall curve

     Uasge:
     Auc of precision recall curve give good indicator of over all model peformance.
     """
     precision, recall, _ = precision_recall_curve(y_score=y_pred_prob,y_true=y_true)
     
     return float(auc(x=recall, y=precision))


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch



if __name__ == "__main__": 
    torch.manual_seed(42)  

    #load data 
    df = load_data(PathEnum.TRAIN_PATH.value) 
    df_val = load_data(PathEnum.VAL_PATH.value) 
    df_test = load_data(PathEnum.TEST_PATH.value) 

    #preprocess data 
    preprocessor = Preprocessing() 
    X_train, y_train, clip_limts, scaler = preprocessor.fit_transform(df)  
    X_val, y_val = preprocessor.transform(df_val, clip_limts, scaler)
    X_test, y_test= preprocessor.transform(df_test, clip_limts, scaler)

    
    model = FraudDetectionNN()
    alpha = 0.75 # (rate to make balance class)                
    gamma = 2 # (focusing on hard samples "minority class") 
    lr = 0.001 

    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer =  torch.optim.SGD(model.parameters(), lr=lr)

    #convert data to tensors
    X_train_tensor = torch.tensor(X_train ,dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    batch_size = 512 # 1024 2048 4096
    num_epochs = 450
    start_epoch = 0

    run_name = f"gamma_{gamma}_alpha_{alpha}_batch_size_{batch_size}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}_SGD_optimizer")

    # Uncomment the line below to load from a checkpoint after training.
    checkpoint_id = 400
    path = f'models/focal_loss_checkpoints/checkpoint_epoch_{checkpoint_id}.pth'
    start_epoch = load_checkpoint(model, path) + 1

    
    # # training loop 
    # for epoch in range(start_epoch, num_epochs):
    #     model.train() 
    #     epoch_loss = 0.0 

    #     # shuffle training data
    #     permutation = torch.randperm(X_train_tensor.size()[0])
    #     X_train_tensor_shuffled = X_train_tensor[permutation].clone()
    #     y_train_tensor_shuffled = y_train_tensor[permutation].clone() 

    #     for i in range(0, len(X_train_tensor), batch_size):
    #         X_batch = X_train_tensor_shuffled[i:i+batch_size]
    #         y_batch = y_train_tensor_shuffled[i:i+batch_size]

    #         optimizer.zero_grad()
    #         output = model(X_batch)
    #         loss = criterion(output, y_batch)
    #         loss.backward()
    #         optimizer.step()

    #         epoch_loss += loss.item()

    #     # log epoch statistics
    #     epoch_loss /= len(X_train_tensor) / batch_size
    #     writer.add_scalar('Loss/train', epoch_loss, epoch)  

    #     for name, param in model.named_parameters():
    #         writer.add_histogram(name, param, epoch)
    #     if param.grad is not None:
    #         writer.add_histogram(f'{name}.grad', param.grad, epoch)

    #     print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, epoch_loss))


    #     model.eval()
    #     with torch.no_grad():
    #         val_output = model(X_val_tensor)
    #         val_loss = criterion(val_output, y_val_tensor).item()
    #         writer.add_scalar('Loss/validation', val_loss, epoch)

    #     # Checkpoint
    #     if (epoch + 1) % 10 == 0:
    #         save_checkpoint(model, epoch + 1, title=run_name)

    #         model.eval()
    #         with torch.no_grad():
    #             val_output = model(X_val_tensor)
    #             val_loss = criterion(val_output, y_val_tensor).item()
        
    #             y_val_prob = val_output.sigmoid().numpy()
    #             y_val_pred = (y_val_prob > 0.5).astype(int)
                
    #             report_val = classification_report(y_true=y_val, y_pred=y_val_pred, output_dict=True)
    #             auc_pr = eval_auc_precision_recall_curve(y_pred_prob=y_val_prob, y_true=y_val)

    #             writer.add_scalar('Validation/Precision',report_val["1"]["precision"], epoch + 1)
    #             writer.add_scalar('Validation/Recall', report_val["1"]["recall"], epoch + 1)
    #             writer.add_scalar('Validation/F1', report_val["1"]["f1-score"], epoch + 1) 
    #             writer.add_scalar('Validation/AUC', auc_pr, epoch + 1)  



    #evaluate model on training and validation data
    model.eval()
    with torch.no_grad():
        val_output = model(X_train_tensor)
        y_train_prob = val_output.sigmoid().numpy()
        y_train_pred = (y_train_prob > 0.5).astype(int)
        _ = eval_classification_report_confusion_matrix(y_true=y_train, y_pred=y_train_pred, title='FraudDetectionNN train') 
        # eval_precision_recall_for_different_threshold(y_pred=y_train_prob, y_true=y_train)

        val_output = model(X_val_tensor)
        y_val_prob = val_output.sigmoid().numpy()
        y_val_pred = (y_val_prob > 0.50).astype(int)
        report_val = eval_classification_report_confusion_matrix(y_true=y_val, y_pred=y_val_pred, title='FraudDetectionNN valdtion')

        
        optimal_threshold, f1_scores = eval_best_threshold(y_pred=y_train_prob, y_true=y_train, with_repect_to="f1_score")  
        y_val_pred = (y_val_prob > optimal_threshold).astype(int)
        report_val = eval_classification_report_confusion_matrix(y_pred=y_val_pred, y_true=y_val, title='FraudDetectionNN optimal threshold')

    writer.close()  # tensorboard --logdir=runs
     

######################################################################################################################################
# Some learning lessons & Notes:
# 1. Alpth and gamma sometimes unstables train using batchnorm make this effect less occur and switching from Adam to SGD also.     
# 2. High gamma (5~7) gives very noisey loss Curve 
# 3. Alpha is very crucial to balance the two classes (need hyperparameter tuning).
#####################################################################################################################################
     

