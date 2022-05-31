import os
from pickletools import optimize
from timeit import default_timer as timer
from datetime import timedelta

# Data processing imports
import pandas as pd

# torch-related imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose
from tripletfolder import TripletLoader
from model_wrap import model_wrap

from fastaudio.augment.signal import ResizeSignal

from config import config


# Progress-bar
from tqdm import tqdm

from utils import * # Just wrap around an iterator (tqdm(range(some_list)))
tqdm.pandas() # Integrate with pandas (for progress_apply())

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

audio_config = AudioConfig.BasicMelSpectrogram(
        sample_rate = config['sample_rate'],
        hop_length = config['hop_length'],
        win_length = config['win_length'],
        n_fft = config['n_fft'],
        n_mels = config['n_mels'],
        normalized = True,
        f_max=config['f_max']
    )

to_spectrum = AudioToSpec.from_cfg(audio_config)

batch_tfms = Compose([
    to_spectrum,
    SpecNormalize(torch.tensor(-43.1299).to(device), torch.tensor(27.4627).to(device)),
])

# Makes all examples the same length. Make sure that the audio files have the same sample rate
# Length is given with (duration/1000)*sample_rate
item_tfms = Compose([
    ResizeSignal(5000)
])

# Using collate is generally a little bit slower, but we have to ensure all audio signal are the same length
def collate_fn(data):
    anchors = torch.stack([item_tfms(x) for x, _,_,_ in data], dim=0)
    targets = torch.stack([item_tfms(x) for _, x,_,_ in data], dim=0)
    pos = torch.stack([item_tfms(x) for _, _,x,_ in data], dim=0)
    neg = torch.stack([item_tfms(x) for _, _,_,x in data], dim=0)
        
    return anchors, targets, pos, neg
    

def load_dataset():
      
    data = get_data(batch_tfms=batch_tfms, 
                sample_rate=config['sample_rate'],
                item_tfms=item_tfms,
                batch_size=config['batch_size'],
                fold=config['fold'],
                path=config['dataset_path'],
                seed=config['seed'], augments=False)
    
    train_loader = torch.utils.data.DataLoader(TripletLoader(data.train),batch_size=config['batch_size'], shuffle=True, num_workers=12, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(TripletLoader(data.valid), batch_size=config['batch_size'], shuffle=True, num_workers=12, collate_fn=collate_fn)
    
    return train_loader, test_loader

train_loader, test_loader = load_dataset()
    
def save_network(network, optimizer, scheduler, epoch, loss, epoch_label, name):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./final_models',name,save_filename)
    if scheduler is not None:
        torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                }, save_path)
    else:
        torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, save_path)
    if torch.cuda.is_available:
        network.to(device)
    
#########################################
# Training the model

threshold = config['validation_threshold']

def evaluate(model, test_loader, loss_fn):
    model.eval()
    outputs = []
    with torch.no_grad():
        for anchor, labels, pos, neg in test_loader:
            anchor, labels, pos, neg = batch_tfms(anchor.to(device)), labels.to(device), batch_tfms(pos.to(device)), batch_tfms(neg.to(device))
            anchor, labels, pos, neg = Variable(anchor), Variable(labels), Variable(pos), Variable(neg)
            outputs.append(model.validation_step((anchor, labels, pos, neg), loss_fn=loss_fn, threshold=threshold))
    return model.validation_epoch_end(outputs)


def train_model(model, criterion, optimizer, scheduler, device, num_epochs=50, start_epoch=0):
    time_start = timer()
    
    history = []
    
    for epoch in range(start_epoch, num_epochs):
        model.train(True)
        
        for anchor, labels, pos, neg in tqdm(train_loader):
            anchor, labels, pos, neg = batch_tfms(anchor.to(device)), labels.to(device), batch_tfms(pos.to(device)), batch_tfms(neg.to(device))
            anchor, labels, pos, neg = Variable(anchor), Variable(labels), Variable(pos), Variable(neg)
            
            # Zero-out the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = model.training_step(anchor, pos, neg, criterion)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        # Validation phase
        result = evaluate(model, test_loader, criterion)
        history.append(result)
        model.epoch_end(epoch, result)
        loss.item()
        if scheduler is not None:
            scheduler.step()
        save_network(model.get_model(), optimizer, scheduler, epoch, loss, str(epoch), 'resnet18_test')
        
    end = timer()
    print(f'Time used for training: {timedelta(seconds=end-time_start)}')
    
def load_network(model, optimizer, scheduler, model_name, model_tag):
    save_path = os.path.join('./final_models',model_name,f'net_{model_tag}.pth')
    
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch+1, loss

arch = config['arch'] == 'resnet34'

resnet = get_resnet(arch, freeze=False)


criterion = nn.TripletMarginLoss(margin=1, swap=True)
optimizer = torch.optim.Adam(params=resnet.parameters(), lr=config['learning_rate'])
optimizer = torch.optim.Adam([
    {'params': resnet.conv1.parameters(), 'lr': 1e-5},
    {'params': resnet.bn1.parameters(), 'lr': 1e-5},
    {'params': resnet.relu.parameters(), 'lr': 1e-5},
    {'params': resnet.maxpool.parameters(), 'lr': 1e-5},
    {'params': resnet.layer1.parameters(), 'lr': 1e-5},
    {'params': resnet.layer2.parameters(), 'lr': 1e-5},
    {'params': resnet.layer3.parameters(), 'lr': 1e-4},
    {'params': resnet.layer4.parameters(), 'lr': 1e-4},
    {'params': resnet.fc.parameters(), 'lr': config['learning_rate']},
    ], lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=1/10) # None #
resnet.to(device)
epoch = 0
#epoch, loss = load_network(resnet, optimizer, None, 'resnet34_esc10_1', '30')

model = model_wrap(resnet, device, distance=config['distance'])

train_model(model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, num_epochs=config['n_epochs'], start_epoch=epoch)
