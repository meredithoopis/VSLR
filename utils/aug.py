import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
from typing import Tuple


df = pd.read_csv('record/cc.csv')

class DataAugmenter: 
    def __init__(self, filepath): 
        self.df = pd.read_csv(filepath)
        self.coords = self.df[['x', 'y']].fillna(0)
        self.coords = self.df[['x', 'y']].to_numpy()
        self.mask = (self.df['x'] != 0) & (self.df['y'] != 0)
    
    def flip(self): 
        mapping = {"right_head": "left_head", 
                   "Right_hand": "Left_hand", 
                   "right_arm": "left_arm", 
                   "left_head": "right_head", 
                   "Left_hand": "Right_hand", 
                   "left_arm": "right_arm"
                   }
        mask = self.df['type'].isin(mapping.keys())
        self.df.loc[mask & self.mask, 'x'] = 1 -self.df.loc[mask & self.mask, 'x']
        self.df.loc[mask, 'type'] = self.df.loc[mask, 'type'].map(mapping)
        self.coords = self.df[['x', 'y']].to_numpy()

    '''def resample(self, rate=(0.8,1.2)):
        new_df = pd.DataFrame()
        for body_part, group in self.df.groupby('type'):
            length = len(group)
            new_size = int(np.random.uniform(rate[0], rate[1]) * length)
            f = interp1d(np.arange(length), group[['x', 'y']].to_numpy(), axis=0)
            new_coordinates = f(np.linspace(0, length-1, new_size))
            new_group = pd.DataFrame(new_coordinates, columns=['x', 'y'])
            new_group['type'] = body_part
            new_df = pd.concat([new_df, new_group])
        self.df = new_df.sort_values(by=['type'])
        self.coords = self.df[['x', 'y']].to_numpy()'''
    
    def add_noise(self, noise_range = (0, 0.005)):
        noise = np.random.normal(*noise_range, size = self.coords.shape)
        self.coords[self.mask] += noise[self.mask]

    
    def random_resize(self,scale_range = (0.8, 1.2)): 
        scale = np.random.uniform(*scale_range)
        self.coords[self.mask] *= scale 
    
    def spatial_affine(self, scale = (0.8, 1.2), shear = (-0.1, 0.1),shift = (-0.1, 0.1), degree = (-30,30)): 
        center = np.array([0.5, 0.5]) #Set the transformation to the middle 
        if scale is not None: #Zoom in or out 
            scale = np.random.uniform(*scale)
            self.coords[self.mask] *= scale 
        if shear is not None: #Distorts (vertical or horizontal )
            shear_x = shear_y = np.random.uniform(*shear)
            if np.random.uniform() < 0.5: 
                shear_x = 0. 
            else: 
                shear_y = 0. 
            
            shear_mat = np.array([[1., shear_x], [shear_y, 1.]])
            xy = self.coords @ shear_mat 
            center += [shear_y, shear_x]
            self.coords = xy 
        if degree is not None: #Rotate
            xy = self.coords - center 
            degree = np.random.uniform(*degree)
            radian = degree / 180 * np.pi 
            rotate_mat = np.array([[np.cos(radian), np.sin(radian)], [-np.sin(radian), np.cos(radian)]])
            xy = xy @ rotate_mat 
            xy += center 
            self.coords = xy 
        if shift is not None: #Shift(horizontal or vertical )
            shift = np.random.uniform(*shift)
            self.coords[self.mask] += shift 
    def center_crop(self, crop_size=0.5): #If crop then the number of data points will drop randomly 
        lb = (1-crop_size) / 2
        ub = (1+crop_size) / 2 
        mask = (self.df['x'] > lb) & (self.df['x'] < ub)
        self.df = self.df[mask]
        self.coords = self.df[['x', 'y']].to_numpy()
    def clip(self): #Clip to (0,1)
        self.coords = np.clip(self.coords, 0,1)

    def save(self, filepath):
        df_augmented = self.df.copy()
        df_augmented[['x', 'y']] = self.coords
        df_augmented.to_csv(filepath, index=False)


        
aug = DataAugmenter('record/cc.csv')
aug.flip()
aug.add_noise()
aug.random_resize()
aug.spatial_affine()
#aug.center_crop()
#aug.clip()
aug.save('utils/cc_aug.csv')

