
from utils import *
from fingerprinter import *
from config import config
from lshashpy3 import LSHash
from lsh import *
import argparse
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('test_type', metavar='T', type=str, default='classify', help='what test to run (classify/similarity)')
parser.add_argument('-n', dest='n', type=int, default=5, help='How many samples should be considered in the top-n score')
parser.add_argument('-u', dest='u', type=bool, default=False, help='Use the UrbanSound8K dataset instead')
parser.add_argument('-r', dest='r', type=bool, default=False, help='Use ResNet34 instead of 18')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_tfms = get_batch_transforms(device)
item_tfms = get_item_transforms()

similarity_classes = ['thunderstorm', 'can_opening', 'dog', 'church_bells', 'pouring_water']

total_total = top1_accuracy_total = top5_accuracy_total = mctop5Accuracy_total = 0
accurate_total = one_off_total = top5_total = 0
for i in range(1, 2):
    
    # Load the model as-is from pytorch hub
    resnet = get_resnet(args.r, True).to(device)

    arch = 'resnet34' if args.r else 'resnet18'

    # Give it to the fingerprinter. It loads the trained model into the resnet model
    f = fingerprinter(resnet, f'{arch}_test', '20', device, batch_tfms, config['distance'])
    lsh = LSHash(1, config['fing_len'], num_hashtables=2, matrices_filename='weights.npz', hashtable_filename='hash.npz', overwrite=True)
    
    if args.test_type == 'classify':
        start = timer()
        
        if args.u:
            data, _, _ = get_urbansound(batch_tfms=batch_tfms, item_tfms=item_tfms,
                        sample_rate=config['sample_rate'],
                        batch_size=config['batch_size'],
                        fold=i,
                        seed=config['seed'], device=device, shuffle=False)
        else:
            data, _, _ = get_dataloader(batch_tfms=batch_tfms, item_tfms=item_tfms,
                        sample_rate=config['sample_rate'],
                        batch_size=config['batch_size'],
                        fold=i,
                        path=config['dataset_path'],
                        seed=config['seed'], device=device, shuffle=False, esc10=False)

        train_loader, test_loader = data.train, data.valid
        
        fill_database(f, lsh, train_loader, batch_tfms, item_tfms)
    
        top1, top5, mctop5, total, top1_accuracy, top5_accuracy, mctop5Accuracy = classify(f, lsh, test_loader, batch_tfms, item_tfms)
        total_total += total
        top1_accuracy_total += top1_accuracy
        top5_accuracy_total += top5_accuracy
        mctop5Accuracy_total += mctop5Accuracy
        print(i, top1, top5, mctop5, total, top1_accuracy, top5_accuracy, mctop5Accuracy)
        print(timer()-start)
        
    elif args.test_type == 'similarity':
        avg_diff = 0
        accurate = 0
        top5 = 0
        one_off = 0
        start = timer()
        for cat in similarity_classes:
            data, df_train, df_test = get_dataloader(batch_tfms=batch_tfms, item_tfms=item_tfms,
                        sample_rate=config['sample_rate'],
                        batch_size=config['batch_size'],
                        fold=i,
                        path=config['dataset_path'],
                        seed=config['seed'], 
                        #category=cat, 
                        device=device, 
                        shuffle=False
                        )

            train_loader, test_loader = data.train, data.valid
            
            fill_database(f, lsh, train_loader, batch_tfms, item_tfms)
            
            similar_sounds = similarity(f, lsh, test_loader, batch_tfms, item_tfms, df_train, df_test, args.n)
            
            df = pd.read_csv(f'../datasets/ESC-50-master/meta/esc-50-{cat}.csv', index_col=0)
            
            for sounds in similar_sounds:
                # get every score from the list of most-similar sounds
                scores = []
                for x in sounds[1]:
                    try:
                        scores.append(df.loc[sounds[0], x])
                    except KeyError as e:
                        continue
                
                # Get the assumed best score
                if len(scores) > 0:
                    top_score = scores[0]
                
                    # Get the actually best available score
                    actual_score = df[sounds[0]].max()
                    
                    # Is this score among the returned values?
                    if actual_score in scores:
                        top5 += 1
                        
                    # Is the assumed best match actually right?
                    if top_score == actual_score:
                        accurate += 1
                    
                    # If not, was it off by just 1?
                    elif top_score == actual_score-1:
                        one_off += 1
                        
                    # How off was it?
                    avg_diff += actual_score - top_score
                    print(sounds[0], sounds[1], top_score, actual_score)
        
        accurate_total += accurate
        one_off_total += one_off
        top5_total += top5
        print(avg_diff/40, accurate/40, (one_off+accurate)/40, top5/40)
        print(timer()-start)
    else:
        print('unsupported test type (test_type)')
        exit()

if args.test_type == 'classify':
    print(total_total, top1_accuracy_total, top5_accuracy_total, mctop5Accuracy_total)
else:
    print(accurate_total, (one_off_total+accurate_total), top5_total)