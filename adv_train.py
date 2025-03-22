"""
File contains a script for an adversarial domain adoptation procedure
"""

import torch
import hydra
import logging
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification
from utils import get_mixed_dataloader, get_dataloaders, set_all_seeds


    
def train_clf_steps(num_steps, model, domain_clf, loader, optimizer, writer, glob_epoch_idx): 
    domain_adv_loss = torch.nn.BCEWithLogitsLoss()
    cls_loss = torch.nn.CrossEntropyLoss()
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)): 
        
        X, y = batch
        X = X.to(model.device)
        #y = y.to(torch.float32)
        domain_labels = torch.zeros_like(y, device = model.device, dtype = torch.float32)
        y = y.to(model.device)
        
        optimizer.zero_grad()
        out_clf =  model(X)
        print(out_clf.keys())
        clf_token_embed = model.vit(X)['last_hidden_state'][:, 0, :]
        pred_domain = domain_clf(clf_token_embed)
        pred_clf = model.classifier(clf_token_embed)
        domain_loss = domain_adv_loss(pred_domain, domain_labels.reshape(-1, 1))    
        clf_loss = cls_loss(pred_clf, y)
        loss = clf_loss + domain_loss
        print(f'Domian_loss = {domain_loss.item()} , CE loss = {clf_loss.item()}')
        loss.backward()
        optimizer.step()    
        
        if batch_idx > num_steps: 
            return 
    
def train_domain_clf_steps(model, domain_clf, loader, optimizer, writer, glob_epoch_idx): 
    domain_adv_loss = torch.nn.BCEWithLogitsLoss()
    
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)): 
        
        X, y = batch
        X = X.to(model.device)
        y = y.to(torch.float32)
        y = y.to(model.device)
        
        optimizer.zero_grad()

        clf_token_embed = model.vit(X)['last_hidden_state'][:, 0, :]
        pred = domain_clf(clf_token_embed)
        loss = domain_adv_loss(pred, y.reshape(-1, 1))    
        
        #print(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx > num_steps: 
            return         
@torch.inference_mode()
def eval_epoch(model, domain_clf, loader_clf, loader_domain_clf,  writer, glob_epoch_idx, test = False): 
    acc = 0
    avg_loss = 0
    for batch_idx, batch in tqdm(enumerate(loader_clf), total=len(loader_clf)): 
        
        X, y = batch
        X = X.to(model.device)
        y = y.to(model.device)
        
        pred = model(pixel_values = X, labels = y)
        class_pred = pred.logits.argmax(axis = 1)
        acc += (class_pred == y).type(torch.float).mean()
        avg_loss += pred.loss.item()

        del pred
        torch.cuda.empty_cache()
        
    if test:
        print(f'Test classification loss : {avg_loss / len(loader):.3f}')
        print(f'Test classification acc : {acc / len(loader):.3f}')
    else: 
        writer.add_scalar("Eval/loss_epoch", avg_loss / len(loader), glob_epoch_idx)
        writer.add_scalar("Eval/acc_epoch",acc / len(loader), glob_epoch_idx) 
        
    for batch_idx, batch in tqdm(enumerate(loader_clf), total=len(loader_clf)): 
        
        X, y = batch
        X = X.to(model.device)
        y = y.to(model.device)
        
        pred = model(pixel_values = X, labels = y)
        class_pred = pred.logits.argmax(axis = 1)
        acc += (class_pred == y).type(torch.float).mean()
        avg_loss += pred.loss.item()

        del pred
        torch.cuda.empty_cache()
        
    if test:
        print(f'Test classification loss : {avg_loss / len(loader):.3f}')
        print(f'Test classification acc : {acc / len(loader):.3f}')
    else: 
        writer.add_scalar("Eval/loss_epoch", avg_loss / len(loader), glob_epoch_idx)
        writer.add_scalar("Eval/acc_epoch",acc / len(loader), glob_epoch_idx) 
    return avg_loss / len(loader),  acc / len(loader)
    
    
@hydra.main(version_base=None, config_path="configs", config_name="adversarial_scheme")    
def run_training(cfg): 
    
    set_all_seeds(cfg.seed)
    train_clf_loader, valid_clf_loader, test_clf_loader, train_domain_clf_loader, valid_domain_clf_loader, test_domain_clf_loader = get_mixed_dataloader(cfg.data)
    
    model = ViTForImageClassification.from_pretrained(cfg.hf_model_path)
    model.load_state_dict(torch.load(cfg.tuned_model_path)['model_state_dict'])
    model = model.to(cfg.device)
    
    logging.info(f'Loaded pretrained VIT')
    

    logging.info(f'Instantiatd classifier optimizer')
    domain_clf = torch.nn.Sequential(
        torch.nn.LazyLinear(64), 
        torch.nn.PReLU(),
        torch.nn.LazyLinear(1),
    )
    domain_clf = domain_clf.to(cfg.device)
    logging.info(f'Instantiated domain classifier')
    optimizer_domain_clf = hydra.utils.instantiate(cfg.optimizer_domain_clf, domain_clf.parameters())
    optimizer_clf = hydra.utils.instantiate(cfg.optimizer_clf, [
                {'params': model.parameters(), 'lr': 3e-4},
                {'params': domain_clf.parameters(), 'lr' : 0}
            ])
    logging.info(f'Instantiated domain classifier optimizer')
    writer = SummaryWriter(cfg.log_path)
    
    logging.info(f'Starting training')
    #train_domain_clf_steps(10000, model, domain_clf, train_clf_loader, optimizer_clf, writer, glob_epoch_idx = 0 )
    
    for glob_epoch_idx in range(cfg.n_epochs):
        logging.info(f'Classification step')
        train_clf_steps(7, model, domain_clf, train_clf_loader, optimizer_clf, writer, glob_epoch_idx = 0 )
        logging.info(f'Domain step')
        train_domain_clf_steps(5, model, domain_clf, train_domain_clf_loader, optimizer_domain_clf, writer, glob_epoch_idx = 0 )
        
        #train_epoch(model, train_loader, optimizer, writer, glob_epoch_idx)
    #    eval_clf_loss, eval_clf_acc, eval_domain_clf_loss, eval_domain_clf_acc = eval_epoch(model, valid_loader, writer, glob_epoch_idx)
        
    #    torch.save({
    #            'epoch': glob_epoch_idx,
    #            'model_state_dict': model.state_dict(),
    #            'domain_clf' : domain_clf.state_dict(), 
    #            'optimizer_state_dict': optimizer.state_dict(),
    #            }, f'{cfg.ckpt_path}/model_epoch_{glob_epoch_idx}_acc_{eval_acc:.3f}.pth')
        
    #eval_epoch(model, test_loader, writer, glob_epoch_idx, test = True)


if __name__ == '__main__': 
    run_training()
