
from utils import *

from tqdm import tqdm

tqdm.pandas()

def fill_database(f, lsh, train_loader, batch_tfms, item_tfms):
    # Fill the database
    c = 0
    for batch, targets in train_loader:
        batch_tfms(batch)
        item_tfms(batch)
        
        fingerprints = f.get_fingerprint_from_batch(batch)
        for fing, target in zip(fingerprints, targets):
            
            # For some reason it stores "None" if the value is a number converted from target.item()
            # Probably something to do with pointers and how python handles moving data 
            # (perhaps it is deleted from memory after the pointer is moved into the lsh?)
            lsh.index(fing.numpy(), extra_data=str(f'{target.item()} {c}'))
            c+=1

def classify(f, lsh, test_loader, batch_tfms, item_tfms):
    def most_common(lst):
        return max(set(lst), key=lst.count)
    
    mt5 = 0
    t5 = 0
    t = 0
    t1 = 0
    for batch, targets in test_loader:
        batch_tfms(batch)
        item_tfms(batch)
        
        hashes = f.get_fingerprint_from_batch(batch)
        for hash, target in zip(hashes, targets):
            
            q_resp = lsh.query(hash.numpy(), num_results=5)
            labels = [int(x[0][1].split(' ')[0]) for x in q_resp]
            if labels[0] == target.item(): # top-1 accuracy
                t1 += 1

            if most_common(labels) == target.item(): # Most common top-5 accuracy
                mt5 += 1
            if target.item() in labels[:5]: # Top-5 accuracy
                t5 += 1
                
            t += 1
            
    return t1, t5, mt5, t, t1/t, t5/t, mt5/t

def similarity(f, lsh, test_loader, batch_tfms, item_tfms, df_train, df_test, n):
    similarity_list = []
    t = 0
    for batch, _ in test_loader:
        batch_tfms(batch)
        item_tfms(batch)
        
        hashes = f.get_fingerprint_from_batch(batch)
        for hash in hashes:
            
            q_resp = lsh.query(hash.numpy(), num_results=n)
            indexes = [int(x[0][1].split(' ')[1]) for x in q_resp]
            
            most_similar = [df_train.iloc[x, 0] for x in indexes]
            
            similarity_list.append((df_test.iloc[t, 0], most_similar))
            
            t += 1
            
    return similarity_list


    
