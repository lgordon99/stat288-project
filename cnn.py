# imports
import numpy as np
import torch

class CNN:
    def __init__(self, setting):
        self.site_dir = utils.get_site_dir(site)
        self.constants = utils.process_yaml('constants.yaml')
        self.classes = ['empty', 'midden', 'mound', 'water']
        self.identifiers = np.load(f'{self.site_dir}/identifiers.npy')
        self.identifier_matrix = np.load(f'{self.site_dir}/identifier_matrix.npy')
        self.labels = np.zeros(len(self.identifiers), dtype=np.int_) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if setting == 'original':
            thermal_tiles = np.load(f'{self.site_dir}/tiles/thermal_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/tiles/rgb_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/tiles/lidar_tiles.npy')
        elif setting == 'png':
            thermal_tiles = np.load(f'{self.site_dir}/png_tiles/thermal_png_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/png_tiles/rgb_png_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/png_tiles/lidar_png_tiles.npy')
        elif setting == 'fuse':
            fused_tiles = np.load(f'{self.site_dir}/fused_tiles.npy')
        elif setting == 'three_channel':
            thermal_tiles = np.load(f'{self.site_dir}/upsampled_tiles/thermal_upsampled_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/upsampled_tiles/grayscale_upsampled_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/upsampled_tiles/lidar_upsampled_tiles.npy')
        elif setting == 'five_channel':
            thermal_tiles = np.load(f'{self.site_dir}/upsampled_tiles/thermal_upsampled_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/upsampled_tiles/rgb_upsampled_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/upsampled_tiles/lidar_upsampled_tiles.npy')
 
        train_identifiers = self.identifier_matrix.T[-55:].T.ravel()
        train_indices = [np.where(self.identifiers == train_identifier)[0][0] for train_identifier in train_identifiers]
        test_identifiers = self.identifier_matrix.T[-55:].T.ravel()
        test_indices = [np.where(self.identifiers == test_identifier)[0][0] for test_identifier in test_identifiers]


