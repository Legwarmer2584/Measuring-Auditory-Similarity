import os
import torch
from torchvision import transforms
from PIL import Image


class fingerprinter:
    def __init__(self, model, model_name: str, model_tag: str, device, transforms: transforms.Compose, distance='euclidean') -> None:
        self.model = model
        self.model_name = model_name
        self.model_tag = model_tag
        self.device = device
        self.preprocess = transforms
        self.distance = distance
        
        self._load_network() # Loads the network from disk
        self.model.eval()

    def _load_network(self):
        save_path = os.path.join('./final_models',self.model_name,f'net_{self.model_tag}.pth')
        
        checkpoint = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def _binary_step(self, batch):
        sign = torch.sign(batch)
        return torch.relu(sign)
    
    def _thresholed_binary_step(self, batch, threshold):
        batch[batch >= threshold] = 1
        batch[batch < threshold] = 0
    
    def get_fingerprint_from_batch(self, batch):
        self.model.eval()
        with torch.no_grad():
            feats = self.model(batch.to(self.device))
        out = feats.cpu()
        return out

    def get_hash_from_file(self, filename):
        img = Image.open(filename).convert('RGB')
        input_tensor = self.preprocess(img)
        return self.get_hash(input_tensor)

    def get_hash(self, input_tensor):
        # Convert image to a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to(self.device)
            self.model.to(self.device)

        with torch.no_grad():
            output = self.model(input_batch).squeeze(0)
            
        return self._binary_step(output).cpu()
    
    def get_hash_hex(self, binary_input):
        hash_bits = ''.join(['1' if it > 0 else '0' for it in binary_input])
        return '{:0{}x}'.format(int(hash_bits, 2), len(hash_bits) // 4)
