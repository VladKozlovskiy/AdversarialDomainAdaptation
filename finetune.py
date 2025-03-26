import os    
import torch
import hydra
import logging
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification
from utils import get_mixed_dataloader, get_dataloaders, set_all_seeds

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
logging.basicConfig(level="DEBUG")
log = logging.getLogger("Logger")
    
def train_epoch(model, loader, optimizer, writer, glob_epoch_idx):
    
    acc = 0
    avg_loss = 0
    
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)): 
        
        X, y = batch
        X = X.to(model.device)
        y = y.to(model.device)
        
        optimizer.zero_grad()
        pred = model(pixel_values = X, labels = y)
        pred.loss.backward()
        
        optimizer.step()
        class_pred = pred.logits.argmax(axis = 1)
        
        acc += (class_pred == y).type(torch.float).mean()
        avg_loss += pred.loss.item()
        
        glob_iter = batch_idx + glob_epoch_idx*len(loader)
        writer.add_scalar("Train/loss_iter", pred.loss.item(),  glob_iter)
        writer.add_scalar("Train/epoch_iter", glob_epoch_idx, glob_iter)
        
        #del pred
        #torch.cuda.empty_cache()
        
    writer.add_scalar("Train/loss_epoch", avg_loss / len(loader), glob_epoch_idx)
    writer.add_scalar("Train/acc_epoch", acc / len(loader), glob_epoch_idx )
      
@torch.inference_mode()
def eval_epoch(model, loader, writer, glob_epoch_idx, test = False): 
    acc = 0
    avg_loss = 0
    losses, preds, labels = [], [], []
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)): 
        
        X, y = batch
        X = X.to(model.device)
        y = y.to(model.device)
        
        pred = model(pixel_values = X, labels = y)
        class_pred = pred.logits.argmax(axis = 1)
        acc += (class_pred == y).type(torch.float).mean()
        avg_loss += pred.loss.item()
        losses.append(pred.loss.item())
        preds.extend(class_pred.cpu().tolist())
        labels.extend(y.cpu().tolist())
        
        #del pred
        #torch.cuda.empty_cache()
        
    clf_loss = sum(losses) / len(losses)
    clf_acc = accuracy_score(labels, preds)
    
    clf_f1 = f1_score(labels, preds, average='macro')
    clf_kappa = cohen_kappa_score(labels, preds)
    
    
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = torch.tensor(np.array(image)).permute(2, 0, 1)
    
    writer.add_scalar(f'Eval/Classification_Loss', clf_loss, glob_epoch_idx)
    writer.add_scalar(f'Eval/Classification_Accuracy', clf_acc, glob_epoch_idx)
    writer.add_scalar(f'Eval/Classification_F1_Score', clf_f1, glob_epoch_idx)
    writer.add_scalar(f'Eval/Classification_Kappa_Score', clf_kappa, glob_epoch_idx)
    writer.add_image(f'Eval/Confusion_Matrix', image, glob_epoch_idx)
    return clf_f1, clf_kappa
    
@hydra.main(version_base=None, config_path="configs", config_name="domain_finetune")    
def run_training(cfg): 

    set_all_seeds(cfg.seed)
    
    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
        
    train_loader, valid_loader, test_loader = get_dataloaders(cfg.data)
    model = ViTForImageClassification.from_pretrained(cfg.hf_model_path)
    model.load_state_dict(torch.load(cfg.tuned_model_path)['model_state_dict'])
    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    writer = SummaryWriter(cfg.log_path)
    max_kappa = 0
    for glob_epoch_idx in range(cfg.n_epochs):
        
        train_epoch(model, train_loader, optimizer, writer, glob_epoch_idx)
        clf_f1, clf_kappa = eval_epoch(model, valid_loader, writer, glob_epoch_idx)
        
        if clf_kappa > max_kappa:
            
            for item in os.listdir(cfg.ckpt_path): 
                    os.remove(f'{cfg.ckpt_path}/{item}')
                
            torch.save({
                    'epoch': glob_epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'{cfg.ckpt_path}/model_epoch_{glob_epoch_idx}_kappa_{clf_kappa:.3f}.pth')
            max_kappa = clf_kappa


if __name__ == '__main__': 
    run_training()