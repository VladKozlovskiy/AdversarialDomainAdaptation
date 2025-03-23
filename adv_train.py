"""
File contains a script for an adversarial domain adoptation procedure
"""

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



def train_clf_steps(model, domain_clf, loader, optimizer, criterion_clf, criterion_domain, num_steps, writer, global_step):
    model.train()
    domain_clf.train()
    
    for batch_idx, (X, y) in enumerate(tqdm(loader, total=min(num_steps, len(loader)))):
        if batch_idx >= num_steps:
            break
        
        X, y = X.to(model.device), y.to(model.device)
        domain_labels = torch.zeros_like(y, device=model.device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        clf_token_embed = model.vit(X)['last_hidden_state'][:, 0, :]
        pred_domain = domain_clf(clf_token_embed)
        pred_clf = model.classifier(clf_token_embed)
        
        domain_loss = criterion_domain(pred_domain, domain_labels.reshape(-1, 1))
        clf_loss = criterion_clf(pred_clf, y)
        loss = clf_loss + 0.3 * domain_loss
        
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Train/Domain_Loss', domain_loss.item(), global_step)
        writer.add_scalar('Train/Classification_Loss', clf_loss.item(), global_step)
        writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
        
        global_step += 1
        
    return global_step

def train_domain_clf_step(model, domain_clf, loader, optimizer, criterion_domain, num_steps, writer, global_step):
    model.eval()
    domain_clf.train()
    
    for batch_idx, (X, y) in enumerate(tqdm(loader, total=min(num_steps, len(loader)))):
        if batch_idx >= num_steps:
            break
        
        X, y = X.to(model.device), y.to(model.device).float()
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            clf_token_embed = model.vit(X)['last_hidden_state'][:, 0, :]
        
        pred = domain_clf(clf_token_embed)
        loss = criterion_domain(pred, y.reshape(-1, 1))
        
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Train/Domain_Classifier_Loss', loss.item(), global_step)
        
        global_step += 1
        
    return global_step
    
 
@torch.inference_mode()
def eval_epoch(model, domain_clf, loader_clf, loader_domain_clf, writer, epoch, prefix='Eval'):
    model.eval()
    domain_clf.eval()
    
    criterion_clf = torch.nn.CrossEntropyLoss()
    criterion_domain = torch.nn.BCEWithLogitsLoss()
    
    clf_losses, clf_preds, clf_labels = [], [], []
    domain_losses, domain_preds, domain_labels = [], [], []
    
    for X, y in loader_clf:
        X, y = X.to(model.device), y.to(model.device)
        
        clf_token_embed = model.vit(X)['last_hidden_state'][:, 0, :]
        pred_clf = model.classifier(clf_token_embed)
        
        loss = criterion_clf(pred_clf, y)
        clf_losses.append(loss.item())
        clf_preds.extend(pred_clf.argmax(dim=1).cpu().numpy())
        clf_labels.extend(y.cpu().numpy())
    
    for X, y in loader_domain_clf:
        X, y = X.to(model.device), y.to(model.device).float()
        
        clf_token_embed = model.vit(X)['last_hidden_state'][:, 0, :]
        pred_domain = domain_clf(clf_token_embed)
        
        loss = criterion_domain(pred_domain, y.reshape(-1, 1))
        domain_losses.append(loss.item())
        domain_preds.extend((pred_domain > 0).float().cpu().numpy())
        domain_labels.extend(y.cpu().numpy())
    
    clf_loss = sum(clf_losses) / len(clf_losses)
    clf_acc = accuracy_score(clf_labels, clf_preds)
    domain_loss = sum(domain_losses) / len(domain_losses)
    domain_acc = accuracy_score(domain_labels, domain_preds)
    
    clf_f1 = f1_score(clf_labels, clf_preds, average='macro')
    clf_kappa = cohen_kappa_score(clf_labels, clf_preds)
    
    
    cm = confusion_matrix(clf_labels, clf_preds)
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
    
    writer.add_scalar(f'{prefix}/Classification_Loss', clf_loss, epoch)
    writer.add_scalar(f'{prefix}/Classification_Accuracy', clf_acc, epoch)
    writer.add_scalar(f'{prefix}/Classification_F1_Score', clf_f1, epoch)
    writer.add_scalar(f'{prefix}/Classification_Kappa_Score', clf_kappa, epoch)
    writer.add_scalar(f'{prefix}/Domain_Loss', domain_loss, epoch)
    writer.add_scalar(f'{prefix}/Domain_Accuracy', domain_acc, epoch)
    writer.add_image(f'{prefix}/Confusion_Matrix', image, epoch)
    
    log.info(f'{prefix} - Clf Loss: {clf_loss:.4f}, Clf Acc: {clf_acc:.4f}, Clf F1: {clf_f1:.4f}, Clf Kappa: {clf_kappa:.4f}')
    log.info(f'{prefix} - Domain Loss: {domain_loss:.4f}, Domain Acc: {domain_acc:.4f}')
    
    plt.close(fig) 
    
    return clf_loss, clf_acc, clf_f1, clf_kappa, domain_loss, domain_acc
    
    
    
@hydra.main(version_base=None, config_path="configs", config_name="adversarial_scheme")    
def run_training(cfg):
    set_all_seeds(cfg.seed)
    
    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)
    
    train_clf_loader, valid_clf_loader, test_clf_loader, train_domain_clf_loader, valid_domain_clf_loader, test_domain_clf_loader = get_mixed_dataloader(cfg.data)
    
    model = ViTForImageClassification.from_pretrained(cfg.hf_model_path)
    model.load_state_dict(torch.load(cfg.tuned_model_path)['model_state_dict'])
    model = model.to(cfg.device)
    
    domain_clf = torch.nn.Sequential(
        torch.nn.LazyLinear(64), 
        torch.nn.PReLU(),
        torch.nn.LazyLinear(1),
    ).to(cfg.device)
    
    optimizer_domain_clf = hydra.utils.instantiate(cfg.optimizer_domain_clf, domain_clf.parameters())
    optimizer_clf = hydra.utils.instantiate(cfg.optimizer_clf, [
        {'params': model.parameters(), 'lr': 3e-4},
        {'params': domain_clf.parameters(), 'lr': 0}
    ])
    
    criterion_clf = torch.nn.CrossEntropyLoss()
    criterion_domain = torch.nn.BCEWithLogitsLoss()
    
    writer = SummaryWriter(cfg.log_path)
    global_step = 0
    
    max_clf_acc = 0
    
    for epoch in range(cfg.n_epochs):
        log.info(f'Epoch {epoch+1}/{cfg.n_epochs}')
        
        log.info('Domain classifier training step')
        global_step = train_domain_clf_step(model, domain_clf, train_domain_clf_loader, optimizer_domain_clf, criterion_domain, cfg.domain_steps, writer, global_step)
        
        log.info('Classification training step')
        global_step = train_clf_steps(model, domain_clf, train_clf_loader, optimizer_clf, criterion_clf, criterion_domain, cfg.clf_steps, writer, global_step)
        
        if (epoch % cfg.eval_period ) == 0:  
            log.info('Evaluation step')
            clf_loss, clf_acc, clf_f1, clf_kappa, domain_loss, domain_acc = eval_epoch(model, domain_clf, valid_clf_loader, valid_domain_clf_loader, writer, epoch, prefix='Validation')

            if clf_acc > max_clf_acc: 

                torch.save({
                    'epoch': global_step,
                    'model_state_dict': model.state_dict(),
                    }, f'{cfg.ckpt_path}/model_epoch_{global_step}_acc_{clf_acc:.3f}.pth')

                max_clf_acc = clf_acc


if __name__ == '__main__': 
    run_training()
