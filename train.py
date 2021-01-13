import os
import torch
import numpy as np
import argparse
import time
import torch.nn.functional as F

from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset_utility import dataset, ToTensor
from dcnet import DCNet

# different fig_type for RAVEN dataset
# center_single, distribute_four, distribute_nine, left_center_single_right_center_single
# up_center_single_down_center_single, in_center_single_out_center_single, in_distribute_four_out_center_single

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='dcnet')
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--fig_type', type=str, default='*') 
parser.add_argument('--dataset', type=str, default='raven')
parser.add_argument('--root', type=str, default='~/dataset/RAVEN')

#parser.add_argument('--fig_type', type=str, default='neutral') 
#parser.add_argument('--dataset', type=str, default='pgm')
#parser.add_argument('--root', type=str, default='~/dataset/PGM')

parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--img_size', type=int, default=96)
parser.add_argument('--workers', type=int, default=16)
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)

tf = transforms.Compose([ToTensor()])    
    
train_set = dataset(args.root, 'train', args.fig_type, args.img_size, tf)
valid_set = dataset(args.root, 'val', args.fig_type, args.img_size, tf)
test_set = dataset(args.root, 'test', args.fig_type, args.img_size, tf)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

save_name = args.model_name + '_' + args.fig_type + '_' + str(args.dim) + '_' + str(args.img_size)

save_path_model = os.path.join(args.dataset, 'models', save_name)    
if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)    
    
save_path_log = os.path.join(args.dataset, 'logs')    
if not os.path.exists(save_path_log):
    os.makedirs(save_path_log)   
    
model = DCNet(dim=args.dim).to(device)    

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

time_now = datetime.now().strftime('%D-%H:%M:%S')      
save_log_name = os.path.join(save_path_log, 'log_{:s}.txt'.format(save_name)) 
with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        args.lr, args.batch_size, args.img_size, time_now))
f.close() 


def contrast_loss(output, target):
    zeros = torch.zeros_like(output)
    zeros.scatter_(1, target.view(-1, 1), 1.0)
        
    return F.binary_cross_entropy_with_logits(output, zeros)


def train(epoch):
    model.train()    
    metrics = {'loss': [], 'correct': [], 'count': []}
    
    train_loader_iter = iter(train_loader)
    for batch_idx in trange(len(train_loader_iter)):
        image, target = next(train_loader_iter)
        
        image = Variable(image, requires_grad=True).to(device)
        target = Variable(target, requires_grad=False).to(device)
        
        predict = model(image)       
        
        loss = contrast_loss(predict, target)
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()      
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        
        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 
               
    
    print ('Training Epoch: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), accuracy))
            
    return metrics


def validate(epoch):
    model.eval()    
    metrics = {'loss': [], 'correct': [], 'count': []}
            
    valid_loader_iter = iter(valid_loader)
    for _ in trange(len(valid_loader_iter)):
        image, target = next(valid_loader_iter)
        
        image = Variable(image, requires_grad=True).to(device)
        target = Variable(target, requires_grad=False).to(device)

        with torch.no_grad():   
            predict = model(image)        

        loss = contrast_loss(predict, target) 
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()       
                            
        metrics['loss'].append(loss.item())
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))

        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Validation Epoch: {:d}/{:d}, Loss: {:.3f}, Accuracy: {:.3f} \n'.format(
                epoch, args.epochs, np.mean(metrics['loss']), accuracy))
            
    return metrics


def test(epoch):
    model.eval()
    metrics = {'correct': [], 'count': []}
            
    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)
        
        image = Variable(image, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)
                
        with torch.no_grad():   
            predict = model(image)    
            
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        
        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        
        accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Testing Epoch: {:d}/{:d}, Accuracy: {:.3f} \n'.format(epoch, args.epochs, accuracy))
            
    return metrics


if __name__ == '__main__':
    for epoch in range(1, args.epochs+1):
        metrics_train = train(epoch)
        metrics_val = validate(epoch)
        metrics_test = test(epoch)

        # Save model
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

        loss_train = np.mean(metrics_train['loss'])
        acc_train = 100 * np.sum(metrics_train['correct']) / np.sum(metrics_train['count']) 
        
        loss_val = np.mean(metrics_val['loss'])
        acc_val = 100 * np.sum(metrics_val['correct']) / np.sum(metrics_val['count']) 

        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count']) 
                
        time_now = datetime.now().strftime('%H:%M:%S')            
        with open(save_log_name, 'a') as f:
            f.write('Epoch {:02d}: Accuracy: {:.3f} ({:.3f}, {:.3f}), Loss: ({:.3f}, {:.3f}), Time: {:s}\n'.format(
                epoch, acc_test, acc_train, acc_val, loss_train, loss_val, time_now))
        f.close() 
        
        scheduler.step()
   
