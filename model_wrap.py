import torch
from scipy.spatial.distance import hamming # For telling us accuracy
from utils import config
from utils import euclidean_distance

class model_wrap():
    def __init__(self, model, device='cpu', distance='euclidean'):
        self.model = model
        self.device = device
        self.distance = distance
        self.seed = ((-1 - 1) * torch.rand(config['fing_len'], config['num_classes'], generator=torch.Generator().manual_seed(42)) + 1).to(self.device)
        
    def __call__(self, input, *args, **kwds):
        return self.model(input)
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self, val):
        self.model.train(val)
        
    def eval(self):
        self.model.eval()
        
    def get_model(self):
        return self.model
    
    # inputs: `batch`
    # three images each
    def training_step(self, anchor, pos, neg, loss_fn):
        if self.distance == 'hamming':
            # We're forcing it to make the signing more similar
            anc_output = self._binary_step(self(anchor))
            anc_output[anc_output==1] = 1.
            anc_output[anc_output==0] = -1.
        else:
            anc_output = self(anchor)
        
        pos_output = self(pos)
        neg_output = self(neg)                # Generate predictions
        
        # Takes: two (supposedly) similar tensors and one (supposedly) dissimilar tensor
        # anchor, positive, negative
        loss = loss_fn(anc_output, pos_output, neg_output)
        
        return loss
    
    def hamming(self, in1, in2):
        n = len(in1)
        return torch.count_nonzero(torch.eq(in1, in2))/n
    
    def accuracy(self, anchor, pos, neg, threshold):
        total = len(anchor)*2
        true_positive = 0
        true_negative = 0
        false_positive=0
        false_negative = 0
        
        for i, j, k in zip(anchor, pos, neg):
            if self.distance == 'hamming':
                i = self._binary_step(i).cpu()
                j = self._binary_step(j).cpu()
                k = self._binary_step(k).cpu()
                
                if hamming(i, j) <= threshold:
                    true_positive += 1
                else:
                    false_negative += 1
                if hamming(i, k) > threshold:
                    true_negative += 1
                else:
                    false_positive += 1
            else:
                if euclidean_distance(i, j) <= threshold:
                    true_positive += 1
                else:
                    false_negative += 1
                if euclidean_distance(i, k) > threshold:
                    true_negative += 1
                else:
                    false_positive += 1
            
        
        return torch.tensor(true_positive, dtype=torch.float), torch.tensor(true_negative, dtype=torch.float), torch.tensor(false_positive, dtype=torch.float), torch.tensor(false_negative, dtype=torch.float), torch.tensor(total, dtype=torch.float), torch.tensor((true_positive+true_negative)/total)
        
    def _binary_step(self, batch):        
        sign = torch.sign(batch)
        return torch.relu(sign)
        
    def validation_step(self, batch, loss_fn, threshold):
        anchor, _, pos, neg = batch 
        anc_output = self(anchor)
        pos_output = self(pos)
        neg_output = self(neg)
        loss = loss_fn(anc_output, pos_output, neg_output)
        
        # Apply seed (seed.dot(row) -> seed @ row)
        """ anc_output = torch.stack([
            self.seed @ row for row in torch.unbind(anc_output, dim=0)
        ], dim=0)
        
        pos_output = torch.stack([
            self.seed @ row for row in torch.unbind(pos_output, dim=0)
        ], dim=0)
        
        neg_output = torch.stack([
            self.seed @ row for row in torch.unbind(neg_output, dim=0)
        ], dim=0) """
        
        tp, tn, fp, fn, tot, acc = self.accuracy(anc_output, pos_output, neg_output, threshold) # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc, 'val_tp': tp, 'val_tn': tn, 'val_fp': fp, 'val_fn': fn, 'val_tot': tot}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        batch_tp = [x['val_tp'] for x in outputs]
        epoch_tp = torch.stack(batch_tp).mean()      # Combine accuracies
        batch_tn = [x['val_tn'] for x in outputs]
        epoch_tn = torch.stack(batch_tn).mean()      # Combine accuracies
        batch_fp = [x['val_fp'] for x in outputs]
        epoch_fp = torch.stack(batch_fp).mean()      # Combine accuracies
        batch_fn = [x['val_fn'] for x in outputs]
        epoch_fn = torch.stack(batch_fn).mean()      # Combine accuracies
        batch_tot = [x['val_tot'] for x in outputs]
        epoch_tot = torch.stack(batch_tot).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'val_tp': epoch_tp.item(), 'val_tn': epoch_tn.item(), 'val_fp': epoch_fp.item(), 'val_fn': epoch_fn.item(), 'val_tot': epoch_tot}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}, val_tp: {:.4f}, val_tn: {:.4f}, val_fp: {:.4f}, val_fn: {:.4f}, val_tot: {:.4f}".format(epoch, result['val_loss'], result['val_acc'], result['val_tp'], result['val_tn'], result['val_fp'], result['val_fn'], result['val_tot']))