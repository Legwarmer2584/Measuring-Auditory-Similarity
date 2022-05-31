
import pandas as pd
import random


random.seed(1)
accepted_classes = ['thunderstorm', 'can_opening', 'dog', 'church_bells', 'pouring_water']

top5total = 0
cTotal = 0
totalTotal = 0
for _ in range(0, 10): # Run it 10 times and average the score
    for i in range(1, 6):
        c = 0
        total = 0
        top5score = 0
        for category in accepted_classes:
            df = pd.read_csv(f'../datasets/ESC-50-master/meta/esc50.csv', index_col=0)
            df_fold = df[(df['fold'] == i) & (df['category'] == category)]
            df = df[(df['fold'] != i)  & (df['category'] == category)]
            
            matrix = pd.read_csv(f'../datasets/ESC-50-master/meta/esc-50-{category}.csv', index_col=0)
            
            top5 = []
            for t in df_fold.iterrows():
                total += 1
                
                # Get 5 samples from the query
                top5 = [df.sample(n=1) for _ in range(0,5)]
                top5 = [x['fold'].to_string().split('\n')[1].split(' ')[0] for x in top5]
                
                scores = []
                for x in top5:
                    try:
                        scores.append(matrix.loc[t[0], x])
                    except KeyError as e:
                        continue
                
                if len(scores) > 0:
                    # Get the score of the first sample
                    score = scores[0]
                    
                    # What is the actual best-available score for the query
                    actual_score = matrix[t[0]].max()
                    
                    if actual_score in scores:
                        top5score += 1

                    if score == actual_score:
                        c+=1
        print(c, total, top5score)
        cTotal+=c
        top5total += top5score
        totalTotal += total

print((cTotal/50)/40, (top5total/50)/40)