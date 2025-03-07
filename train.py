import torch
import hydra
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification
from utils import get_dataloaders, set_all_seeds


    
    
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
        
        del pred
        torch.cuda.empty_cache()
    writer.add_scalar("Train/loss_epoch", avg_loss / len(loader), glob_epoch_idx)
    writer.add_scalar("Train/acc_epoch", acc / len(loader), glob_epoch_idx )
      
@torch.inference_mode()
def eval_epoch(model, loader, writer, glob_epoch_idx): 
    acc = 0
    avg_loss = 0
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)): 
        
        X, y = batch
        X = X.to(model.device)
        y = y.to(model.device)
        
        pred = model(pixel_values = X, labels = y)
        class_pred = pred.logits.argmax(axis = 1)
        acc += (class_pred == y).type(torch.float).mean()
        avg_loss += pred.loss.item()

        del pred
        torch.cuda.empty_cache()
        
    writer.add_scalar("Eval/loss_epoch", avg_loss / len(loader), glob_epoch_idx)
    writer.add_scalar("Eval/acc_epoch",acc / len(loader), glob_epoch_idx)    

@hydra.main(version_base=None, config_path="configs", config_name="main")    
def run_training(cfg): 
    set_all_seeds(cfg.seed)
    train_loader, valid_loader, test_loader = get_dataloaders(cfg.data)
    model = ViTForImageClassification.from_pretrained(cfg.model_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay = 0.01)
    writer = SummaryWriter(cfg.log_path)
    for glob_epoch_idx in range(cfg.n_epochs): 
        train_epoch(model, train_loader, optimizer, writer, glob_epoch_idx)
        eval_epoch(model, val_loader, writer, glob_epoch_idx)
        
        torch.save({
                'epoch': glob_epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{cfg.ckpt_path}_epoch_{glob_epoch_idx}_acc_{eval_acc:3f}.pth')
        
    eval_epoch(model, test_loader, optimizer, writer, glob_epoch_idx)


if __name__ == '__main__': 
    run_training()