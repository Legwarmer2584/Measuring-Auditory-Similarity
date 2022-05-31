from fastai.vision.all import *
from fastaudio.core.all import *
from torchvision.transforms import Compose
from config import config

def get_resnet(ResNet34: bool, freeze: bool):
    arch = 'resnet34' if ResNet34 else 'resnet18'

    resnet = torch.hub.load('pytorch/vision:v0.10.0', arch, pretrained=config['pretrained'])

    update_first_layer(resnet, n_in=config['n_in'], pretrained=config['pretrained'])

    resnet.fc = nn.Sequential(
        nn.Linear(512, config['num_classes']),
    )
    
    if freeze:
        for name, param in resnet.named_parameters():
            # We only want to freeze the conv layers. downsample.0 is a conv layer, but the name doesn't show up :/
            if 'conv' in name or 'downsample.0.weight' in name:
                param.requires_grad = False
    
    return resnet

def add_augments(df):
    suffix =  ['pps', 'nps', 'sts', 'fts', 'trim'] #

    tmp = df.copy()
    for s in suffix:
        df_cp = df.copy()
        df_cp['filename'] = [f'{x.split(".")[0]}_{s}.wav' for x in df_cp['filename']]
        tmp = pd.concat((tmp, df_cp), ignore_index=True)
    return tmp

# As per https://fastaudio.github.io/ESC50:%20Environmental%20Sound%20Classification/
def get_data(sample_rate=16000, 
             item_tfms=None, 
             batch_tfms=None, 
             fold=1,
             batch_size=32,
             path='./ESC-50-master',
             seed=1,
             device='cuda', esc10=False, augments=False):
    set_seed(seed, True)
    df = pd.read_csv(f'{path}/meta/esc50.csv')
    
    if esc10:
        df = df[df['esc10'] == True]
        
    if augments:
        df = add_augments(df)

    df.reset_index(drop=True, inplace=True) # The library down the line uses loc instead of iloc, so we have to reset the index
    
    splitter = IndexSplitter(df[(df.fold == fold) & ('_' not in df.filename)].index)
    
    audio_block = AudioBlock(sample_rate=sample_rate)
    data_block = DataBlock(
        blocks=(audio_block, CategoryBlock),
        get_x=ColReader('filename', pref=f'{path}/audio/'),
        get_y=ColReader('category'),
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        )
    
    data = data_block.datasets(df)
    return data

def euclidean_distance(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum().sqrt()

def get_urbansound(sample_rate=16000, 
             item_tfms=None, 
             batch_tfms=None, 
             fold=1,
             batch_size=32,
             path='../datasets/UrbanSound8K',
             seed=1,
             category='',
             device='cuda', shuffle=True):
    
    set_seed(seed, True)
    df = pd.read_csv(f'{path}/UrbanSound8K.csv')
        
    df.reset_index(drop=True, inplace=True) # The library down the line uses loc instead of iloc, so we have to reset the index
    
    splitter = IndexSplitter(df[(df.fold == fold) & ('_' not in df['slice_file_name'])].index)
    
    audio_block = AudioBlock(sample_rate=sample_rate)
    data_block = DataBlock(
        blocks=(audio_block, CategoryBlock),
        get_x=ColReader('slice_file_name', pref=f'{path}/audio/merged/'),
        get_y=ColReader('class'),
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        )
    
    data = data_block.dataloaders(df, bs=batch_size, device=device)
    return data, df[df.fold != fold], df[(df.fold == fold) & ('_' not in df['slice_file_name'])]

def get_dataloader(sample_rate=16000, 
             item_tfms=None, 
             batch_tfms=None, 
             fold=1,
             batch_size=32,
             path='./ESC-50-master',
             seed=1,
             category='',
             device='cuda', shuffle=True, esc10=False):
    set_seed(seed, True)
    df = pd.read_csv(f'{path}/meta/esc50.csv')
    
    if esc10:
        df = df[df['esc10'] == True]
    
    # Only look at a single class
    if category != '':
        df = df[df['category'] == category]
        
    df.reset_index(drop=True, inplace=True) # The library down the line uses loc instead of iloc, so we have to reset the index
    
    splitter = IndexSplitter(df[(df.fold == fold) & ('_' not in df['filename'])].index)
    
    audio_block = AudioBlock(sample_rate=sample_rate)
    data_block = DataBlock(
        blocks=(audio_block, CategoryBlock),
        get_x=ColReader('filename', pref=f'{path}/audio/'),
        get_y=ColReader('category'),
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        )
    
    data = data_block.dataloaders(df, bs=batch_size, device=device)
    return data, df[df.fold != fold], df[(df.fold == fold) & ('_' not in df['filename'])]

def get_learner(data, arch, n_channels=1, pretrained=True, normalize=True):
    return cnn_learner(data, arch,
                       n_in=n_channels,
                       pretrained=pretrained,
                       normalize=normalize,
                       loss_func=CrossEntropyLossFlat(), 
                       metrics=accuracy).to_fp16()
    
# courtesy of Chris Kroenke @clck10
# https://enzokro.dev/spectrogram_normalizations/2020/09/10/Normalizing-spectrograms-for-deep-learning.html
class SpecNormalize(Normalize):
    # Normalize/denorm batch of `TensorImage`
    def encodes(self, x:TensorImageBase): return (x-self.mean) / self.std
    def decodes(self, x:TensorImageBase):
        f = to_cpu if x.device.type=='cpu' else noop
        return (x*f(self.std) + f(self.mean))
    
class StatsRecorder:
    def __init__(self, red_dims=(0,2,3)):
        """Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError('Data dims do not match previous observations.')
            
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n
            
def get_first_layer(m):
    "Access first layer of a model"
    c,p,n = m,None,None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p,c=c,getattr(c,n)
    return c,p,n

def load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    if n_in==1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
    elif n_in==2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:,:2] * 1.5
    else:
        # keep 3 channels weights and set others to null
        new_layer.weight.data[:,:3] = previous_layer.weight.data
        new_layer.weight.data[:,3:].zero_()

def update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3: return
    first_layer, parent, name = get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr:getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)
    
def get_batch_transforms(device):
    audio_config = AudioConfig.BasicMelSpectrogram(
            sample_rate = config['sample_rate'],
            hop_length = config['hop_length'],
            win_length = config['win_length'],
            n_fft = config['n_fft'],
            n_mels = config['n_mels'],
            normalized = True,
            f_max=config['f_max'],
        )

    to_spectrum = AudioToSpec.from_cfg(audio_config)

    return Compose([
        to_spectrum,
        SpecNormalize(torch.tensor(-43.1299).to(device), torch.tensor(27.4627).to(device)),
    ])
    
def get_item_transforms():
    return Compose([
        ResizeSignal(5000)
    ])