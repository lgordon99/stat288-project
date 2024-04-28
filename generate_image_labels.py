# imports
import numpy as np
import pandas as pd
import sys # change to from sys import argv
import utils

class GenerateImageLabels:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.constants = utils.process_yaml('constants.yaml')

        self.generate_midden_labels()

    def generate_midden_labels(self):
        midden_coordinates_meters = pd.read_csv(f'{self.site_dir}/midden_coordinates_meters.csv').to_numpy().T
        midden_coordinates_pixels_x = (midden_coordinates_meters[0] - self.constants['tight_bounds_meters']['left']) / self.constants['rgb_res_x']
        midden_coordinates_pixels_y = (midden_coordinates_meters[1] - self.constants['tight_bounds_meters']['top']) / self.constants['rgb_res_y']
        midden_coordinates_pixels = np.append(midden_coordinates_pixels_x, midden_coordinates_pixels_y)
        print(midden_coordinates_pixels.shape)

if __name__ == '__main__':
    GenerateImageLabels(site=sys.argv[1])
