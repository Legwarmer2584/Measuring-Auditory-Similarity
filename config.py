
config = dict(
    # spectrum
    sample_rate=44100,
    n_fft=4096,
    n_mels=224,
    hop_length=308,
    win_length=2205,
    f_max=18000,
    
    # model
    arch='resnet18',            # Which model architecture to use. Must be in pytorch' repo
    n_in = 1,                   # Number of channels of the input (Normally just 1. If loading images: 3 (RGB) unless grayscale)
    pretrained = True,          # Use the pretrained resnet18 model
    distance = 'euclidean',       # Changes how the model is trained. 'hamming' for binary output, 'euclidean' for floats
    checkpoint_path = './model/resnet18_binary',
    
    #training
    learning_rate = 1e-3,       # The model is trained more slowly, but this is how fast the user-defined layers train
    n_epochs = 40,              # Number of epochs to train. Saves every epoch, so just set it high
    batch_size=16,              # Limited by your memory
    normalize = True,           # Will the input data be normalized by us?
    num_classes = 128,          # Number of output features
    validation_threshold = 5.,  # How short a distance does the output have to be to be considered similar (just for training)
    
    # data
    seed = 1,
    fold = 1,
    dataset_path = '../datasets/ESC-50-master', #'../datasets/UrbanSound8K',
    dataset_csv = '../datasets/UrbanSound8K/UrbanSound8K.csv',
    audio_path = '/audio/merged/',
    focus_class = 0,
    
    # Fingerprinter
    fing_len = 128,
    augment_search = False
)