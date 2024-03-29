import pandas as pd 
import numpy as np


def add_noise(df):
    new_x = []
    new_y = []
    for x in df['x'].values:
        if x:
            new_x.append(x + np.random.normal(0,0.005))
        else:
            new_x.append(x)
            
    for y in df['y'].values:
        if y:
            new_y.append(y + np.random.normal(0,0.005))
        else:
            new_y.append(y)
            
    df['x'] = new_x
    df['y'] = new_y
            
    return df

if __name__ == "__main__":
    df = pd.read_csv('record/cc1.csv')
    df2 = add_noise(df)
    df2.to_csv('demo.csv')