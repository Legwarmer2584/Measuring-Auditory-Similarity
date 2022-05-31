import torch
from utils import *

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
])

item_tfms = Compose([
    ResizeSignal(5000)
])

for i in range(1, 6):
    data = get_data(batch_tfms=batch_tfms, 
                    sample_rate=config['sample_rate'],
                    item_tfms=item_tfms,
                    batch_size=config['batch_size'],
                    fold=i,
                    path=config['dataset_path'],
                    seed=config['seed'])

    # Getting the normal of the train dataset
    stats = StatsRecorder()
    with torch.no_grad():
        for i in range(0, len(data.train), int(len(data.train)/10)):
            specs = torch.stack([x[0] for x in data.train[i:i+int(len(data.train)/10)]])
            stats.update(to_spectrum(specs))

    print(f"Take the actual values and use those\nMean: {stats.mean}, Std: {stats.std}")