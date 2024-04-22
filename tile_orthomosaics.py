# imports
from cv2 import imread
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import utils

class TileOrthomosaics:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.constants = utils.process_yaml('constants.yaml')

        # create tiles
        self.thermal_tiles = self.tile('thermal')
        print('Created thermal tiles')
        self.rgb_tiles = self.tile('rgb')
        print('Created RGB tiles')
        self.lidar_tiles = self.tile('lidar')
        print('Created LiDAR tiles')

        self.identifiers = list(range(len(self.thermal_tiles)))
        self.max_pixel_vals = [np.max(thermal_tile) for thermal_tile in self.thermal_tiles]
        
        self.remove_empty_tiles()
        print('Removed empty tiles')

        # save tiles
        os.makedirs(f'{self.site_dir}/tiles', exist_ok=True)
        np.save(f'{self.site_dir}/tiles/thermal_tiles', self.thermal_tiles)
        np.save(f'{self.site_dir}/tiles/rgb_tiles', self.rgb_tiles)
        np.save(f'{self.site_dir}/tiles/lidar_tiles', self.lidar_tiles)
        print('Tiles saved')

        # create PNG tiles
        self.png_tile(self.thermal_tiles, 'thermal')
        print('Created thermal PNG tiles')
        self.png_tile(self.rgb_tiles, 'rgb')
        print('Created RGB PNG tiles')
        self.png_tile(self.lidar_tiles, 'lidar')
        print('Created LiDAR PNG tiles')

    def tile(self, modality):
        band = np.load(f'{self.site_dir}/bands/{modality}_bands.npy') if modality == 'rgb' else np.load(f'{self.site_dir}/bands/{modality}_band.npy') 
        interval = self.constants[f'{modality}_interval']
        tiles = []

        for top in range(0, band.shape[0], interval):
            for left in range(0, band.shape[1], interval):
                tiles.append(band[top : top + interval, left : left + interval])

        return tiles

    def remove_empty_tiles(self):
        for i in reversed(range(len(self.identifiers))):
            if np.all(self.thermal_tiles[i] == 0) or np.all(self.rgb_tiles[i] == 0) or np.all(self.lidar_tiles[i] == 0):
                del self.thermal_tiles[i]
                del self.rgb_tiles[i]
                del self.lidar_tiles[i]
                del self.identifiers[i]
                del self.max_pixel_vals[i]

    def png_tile(self, tiles, modality):
        png_tiles = []

        for i in range(len(tiles)):
            plt.figure(dpi=60.7) # dpi=60.7 to get resultant arrays of (224,224,3)
            image = plt.imshow(tiles[i]) # plot the array of pixel values as an image

            if modality == 'thermal' or modality == 'lidar':
                image.set_cmap('inferno')
            
            plt.axis('off') # remove axes
            os.makedirs(f'{self.site_dir}/png_images/{modality}', exist_ok=True)
            plt.savefig(f'{self.site_dir}/png_images/{modality}/{modality}_png_image_{self.identifiers[i]}.png', bbox_inches='tight', pad_inches=0) # temporarily save the image
            plt.close() # close the image to save memory
            png_tiles.append(imread(f'{self.site_dir}/png_images/{modality}/{modality}_png_image_{self.identifiers[i]}.png')) # convert the PNG image to a 3D array

        os.makedirs(f'{self.site_dir}/png_tiles/{modality}', exist_ok=True)
        np.save(f'{self.site_dir}/png_tiles/{modality}_png_tiles', png_tiles)

if __name__ == '__main__':
    TileOrthomosaics(site=sys.argv[1])
