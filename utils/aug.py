import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
from typing import Tuple


df = pd.read_csv('record/cc.csv')

class DataAugmenter: 
    def __init__(self, filepath, outfolder = None): 
        if outfolder:
            self.outfolder = outfolder
        else:
            self.outfolder = "data/augmentation"
        self.filename = filepath.split('/')[-1]
        
        self.df = pd.read_csv(filepath)
        self.coords = self.df[['x', 'y']].fillna(0)
        self.coords = self.df[['x', 'y']].to_numpy()
        self.mask = (self.df['x'] != 0) & (self.df['y'] != 0)
        
    def get_coords(self):
        return self.coords
    
    def get_mask(self):
        return self.mask
    
    def flip(self): 
        mapping = { "right_head": "left_head", 
                    "right_hand": "left_hand", 
                    "right_arm": "left_arm", 
                    "left_head": "right_head", 
                    "left_hand": "right_hand", 
                    "left_arm": "right_arm"
                    }
        
        self.df.loc[self.mask, 'x'] = - self.df.loc[self.mask, 'x']
        self.df.loc['type'] = self.df.loc['type'].map(mapping)

        self.coords = self.df[['x', 'y']].to_numpy()
        
        self._save("flip")
    
    def add_noise(self, noise_range = (0, 0.01)):
        noise = np.random.normal(*noise_range, size = self.coords.shape)
        self.coords[self.mask] += noise[self.mask]
        
        self._save("noise")

    
    def random_resize(self,scale_range = (0.8, 1.2)): 
        scale = np.random.uniform(*scale_range)
        self.coords[self.mask] *= scale 
        
        self._save("random_resize")
    
    def spatial_affine(self, scale = (0.8, 1.2), shear = (-0.1, 0.1),shift = (-0.1, 0.1), degree = (-30,30)): 
        center = np.array([0.5, 0.5]) # Set the transformation to the middle 
        if scale is not None: # Zoom in or out 
            scale = np.random.uniform(*scale)
            self.coords[self.mask] *= scale 
        if shear is not None: # Distorts (vertical or horizontal )
            shear_x = shear_y = np.random.uniform(*shear)
            if np.random.uniform() < 0.5: 
                shear_x = 0. 
            else: 
                shear_y = 0. 
            
            shear_mat = np.array([[1., shear_x], [shear_y, 1.]])
            xy = self.coords @ shear_mat 
            center += [shear_y, shear_x]
            self.coords = xy 
            
        if degree is not None: # Rotate
            xy = self.coords - center 
            degree = np.random.uniform(*degree)
            radian = degree / 180 * np.pi 
            rotate_mat = np.array([[np.cos(radian), np.sin(radian)], [-np.sin(radian), np.cos(radian)]])
            xy = xy @ rotate_mat 
            xy += center 
            self.coords = xy 
            
        if shift is not None: # Shift (horizontal or vertical)
            shift = np.random.uniform(*shift)
            self.coords[self.mask] += shift 
            
        self._save("spatial_affine")
            
    def center_crop(self, crop_size=0.5): # If crop then the number of data points will drop randomly 
        lb = (1-crop_size) / 2
        ub = (1+crop_size) / 2 
        mask = (self.df['x'] > lb) & (self.df['x'] < ub)
        self.df = self.df[mask]
        self.coords = self.df[['x', 'y']].to_numpy()
        
        self.save("center_crop")
        
    def clip(self): # Clip to (0,1)
        self.coords = np.clip(self.coords, 0,1)
        
        self._save("clip")

    def _save(self, task):
        df_augmented = self.df.copy()
        df_augmented[['x', 'y']] = self.coords
        outdir = self.outfolder + "/" + task + "/"
        df_augmented.to_csv(outdir + self.filename, index=False)


if __name__ == "__main__":    
    aug = DataAugmenter('record/cc.csv')
    aug.flip()
    aug.add_noise()
    aug.random_resize()
    aug.spatial_affine()
    #aug.center_crop()
    #aug.clip()

