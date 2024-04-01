import numpy as np 
import pandas as pd 
from scipy.stats import norm
from functions import normalize_ignore_nan



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
        self.is_augment = False
        
    def get_coords(self):
        return self.coords
    
    def get_mask(self):
        return self.mask
    
    def flip(self): 
        self.is_augment = True
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
    
    def add_noise(self, noise_range = (0, 0.01), random_save = 0.1):
        if self.is_augment:
            is_save = np.random.random()
            if is_save < random_save:
                self._save("random_save", task = "noise")
        self.is_augment = True
        noise = np.random.normal(*noise_range, size = self.coords.shape)
        self.coords[self.mask] += noise[self.mask]
        self.coords[self.mask] -= noise_range[0]
        self.coords[self.mask] /= (1 + noise_range[1])

    def renormalize(self):
        self.coords[:,0] = normalize_ignore_nan(self.coords[:,0])
        self.coords[:,1] = normalize_ignore_nan(self.coords[:,1])
    
    def random_resize(self, scale_range = (0.9, 1.1), random_save = 0.1): 
        scale = np.random.uniform(*scale_range)
        self.coords[self.mask] *= scale 
        self.renormalize()
        
        self._save("random_resize")
    
    def spatial_affine(self, scale = (0.8, 1.2), shear = (-0.1, 0.1), shift = (-0.1, 0.1), degree = (-30,30)): 
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
            
        self.renormalize()
        
        self._save("full_pipeline_spatial_affine")
            
    def crop_proba(self, percentage = 0.95, mode = 'center'): # If crop then the number of data points will drop randomly 
        alpha = 1 - percentage
        if mode == 'center':
            upper_prob = alpha/2
            lower_prob = alpha/2
        else:
            upper_prob = np.random.random() * alpha
            lower_prob = alpha - upper_prob
        # print(upper_prob, lower_prob)
            
        lb = -norm.ppf(1 - lower_prob)
        ub = norm.ppf(1 - upper_prob)
        mask = (self.coords > lb) & (self.coords < ub) 
        self.coords[~mask] = np.nan
        
        for i in range(len(self.coords)):
            if np.isnan(self.coords[i][0]) or np.isnan(self.coords[i][1]):
                self.coords[i][0] = np.nan
                self.coords[i][1] = np.nan
        
        self.renormalize()
        
        self._save("full_pipeline_center_crop")
        

    def _save(self, dir, data = None, task = ''):
        df_augmented = self.df.copy()
        df_augmented[['x', 'y']] = self.coords
        outdir = self.outfolder + "/" + dir + "/"
        
        if data:
            data.to_csv(outdir + task + self.filename, index=False)
        else:
            df_augmented.to_csv(outdir + task + self.filename, index=False)
    


if __name__ == "__main__":   
    import os
    # print(os.getcwd()) 
    aug = DataAugmenter('record/half.csv')
    aug.spatial_affine()
    # if np.random.random() < 0.7: 
    #     aug.flip()
    # if np.random.random() < 0.6: 
    #     aug.add_noise()
        
    # save_decision = np.random.random()
    # if save_decision < 0.25:
    #     aug.random_resize()
    # elif save_decision < 0.5:
    #     aug.random_resize()
    # elif save_decision < 0.75:
    #     aug.spatial_affine()
    # else:
    #     aug.random_resize()


