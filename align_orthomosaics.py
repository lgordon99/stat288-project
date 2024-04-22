# imports
import numpy as np
import os
import rasterio
import sys
import utils
import yaml

class AlignOrthomosaics:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.interval_meters = 20

        with rasterio.open(f'{self.site_dir}/tiffs/thermal.tif') as thermal_tiff:
            thermal_tiff_metadata = thermal_tiff.meta
            self.thermal_res_x, _, self.thermal_left_meters, _, self.thermal_res_y, self.thermal_top_meters, _, _, _ = thermal_tiff.transform
            self.thermal_band = self.get_thermal_band(thermal_tiff)
            print(f'Initial thermal band shape = {self.thermal_band.shape}')

        with rasterio.open(f'{self.site_dir}/tiffs/rgb.tif') as rgb_tiff:
            rgb_tiff_metadata = rgb_tiff.meta
            self.rgb_res_x, _, self.rgb_left_meters, _, self.rgb_res_y, self.rgb_top_meters, _, _, _ = rgb_tiff.transform
            self.rgb_bands = self.get_rgb_bands(rgb_tiff)
            print(f'Initial RGB bands shape = {self.rgb_bands.shape}')

        with rasterio.open(f'{self.site_dir}/tiffs/lidar.tif') as lidar_tiff:
            lidar_tiff_metadata = lidar_tiff.meta
            self.lidar_res_x, _, self.lidar_left_meters, _, self.lidar_res_y, self.lidar_top_meters, _, _, _ = lidar_tiff.transform
            self.lidar_res_x = round(self.lidar_res_x, 2)
            self.lidar_res_y = round(self.lidar_res_y, 2)
            self.lidar_band = self.get_lidar_band(lidar_tiff)
            print(f'Initial LiDAR band shape = {self.lidar_band.shape}')

        self.thermal_interval = int(self.interval_meters / self.thermal_res_x)
        self.rgb_interval = int(self.interval_meters / self.rgb_res_x)
        self.lidar_interval = int(self.interval_meters / self.lidar_res_x)

        self.tight_bound_top_meters, self.tight_bound_bottom_meters, self.tight_bound_left_meters, self.tight_bound_right_meters = self.get_tight_bounds()

        self.crop_to_tight_bounds()
        print(f'Post-cropping thermal band shape = {self.thermal_band.shape}')
        print(f'Post-cropping RGB bands shape = {self.rgb_bands.shape}')
        print(f'Post-cropping LiDAR band shape = {self.lidar_band.shape}')

        self.pad_for_divisibility_by_interval()
        print(f'Post-padding thermal band shape = {self.thermal_band.shape}')
        print(f'Post-padding RGB bands shape = {self.rgb_bands.shape}')
        print(f'Post-padding LiDAR band shape = {self.lidar_band.shape}')

        # save bands
        os.makedirs(f'{self.site_dir}/bands', exist_ok=True)
        np.save(f'{self.site_dir}/bands/thermal_band', self.thermal_band)
        print('Thermal band saved as numpy array')
        np.save(f'{self.site_dir}/bands/rgb_bands', self.rgb_bands)
        print('RGB bands saved as numpy array')
        np.save(f'{self.site_dir}/bands/lidar_band', self.lidar_band)
        print('LiDAR band saved as numpy array')

        # save bands as tiffs
        self.array_to_tiff(self.thermal_band, thermal_tiff_metadata, [self.thermal_res_x, self.thermal_res_y], 'thermal')
        self.array_to_tiff(self.rgb_bands, rgb_tiff_metadata, [self.rgb_res_x, self.rgb_res_y], 'rgb')
        self.array_to_tiff(self.lidar_band, lidar_tiff_metadata, [self.lidar_res_x, self.lidar_res_y], 'lidar')
        print('Bands saved as tiffs')

        # save constants
        with open('constants.yaml', 'w') as yaml_file:
            yaml.dump({'thermal_res_x': self.thermal_res_x, 'thermal_res_y': self.thermal_res_y, 'thermal_interval': self.thermal_interval,
                       'rgb_res_x': self.rgb_res_x, 'rgb_res_y': self.rgb_res_y, 'rgb_interval': self.rgb_interval,
                       'lidar_res_x': self.lidar_res_x, 'lidar_res_y': self.lidar_res_y, 'lidar_interval': self.lidar_interval,
                       'num_cols_in_tiling': int(self.thermal_band.shape[1]/self.thermal_interval)},
                      yaml_file,
                      default_flow_style=False)
        print('Constants saved as YAML')

    def get_thermal_band(self, thermal_tiff):
        thermal_band = thermal_tiff.read(4) # 4th band holds the thermal data
        background_value = 2000 # the thermal data is above the background value
        thermal_band[thermal_band <= background_value] = 0 # sets all background pixels to 0
        min_non_background_value = np.min(thermal_band[thermal_band > 0])
        thermal_band[thermal_band > 0] -= min_non_background_value-1 # downshifts the thermal values such that their minimum is 1

        return thermal_band

    def get_rgb_bands(self, rgb_tiff):
        rgb_bands = rgb_tiff.read().transpose(1, 2, 0)

        return rgb_bands

    def get_lidar_band(self, lidar_tiff):
        lidar_band = lidar_tiff.read(1)
        lidar_band[lidar_band < 0] = 0

        return lidar_band

    def get_tight_bounds(self):
        # nonzero bounds in pixels
        thermal_bound_top_pixels, thermal_bound_bottom_pixels, thermal_bound_left_pixels, thermal_bound_right_pixels = utils.get_nonzero_bounds(array=self.thermal_band)
        rgb_bound_top_pixels, rgb_bound_bottom_pixels, rgb_bound_left_pixels, rgb_bound_right_pixels = utils.get_nonzero_bounds(array=self.rgb_bands)
        lidar_bound_top_pixels, lidar_bound_bottom_pixels, lidar_bound_left_pixels, lidar_bound_right_pixels = utils.get_nonzero_bounds(array=self.lidar_band)

        # nonzero bounds in meters
        thermal_bound_top_meters, thermal_bound_bottom_meters, thermal_bound_left_meters, thermal_bound_right_meters = utils.get_bounds_meters(bounds_pixels=[thermal_bound_top_pixels, thermal_bound_bottom_pixels, thermal_bound_left_pixels, thermal_bound_right_pixels],
                                                                                                                                               origin=[self.thermal_left_meters, self.thermal_top_meters],
                                                                                                                                               res=[self.thermal_res_x, self.thermal_res_y])
        rgb_bound_top_meters, rgb_bound_bottom_meters, rgb_bound_left_meters, rgb_bound_right_meters = utils.get_bounds_meters(bounds_pixels=[rgb_bound_top_pixels, rgb_bound_bottom_pixels, rgb_bound_left_pixels, rgb_bound_right_pixels],
                                                                                                                               origin=[self.rgb_left_meters, self.rgb_top_meters],
                                                                                                                               res=[self.rgb_res_x, self.rgb_res_y])
        lidar_bound_top_meters, lidar_bound_bottom_meters, lidar_bound_left_meters, lidar_bound_right_meters = utils.get_bounds_meters(bounds_pixels=[lidar_bound_top_pixels, lidar_bound_bottom_pixels, lidar_bound_left_pixels, lidar_bound_right_pixels],
                                                                                                                                       origin=[self.lidar_left_meters, self.lidar_top_meters],
                                                                                                                                       res=[self.lidar_res_x, self.lidar_res_y])
        # tight bounds in meters
        tight_bound_top_meters = np.min([thermal_bound_top_meters, rgb_bound_top_meters, lidar_bound_top_meters])
        tight_bound_bottom_meters = np.max([thermal_bound_bottom_meters, rgb_bound_bottom_meters, lidar_bound_bottom_meters])
        tight_bound_left_meters = np.max([thermal_bound_left_meters, rgb_bound_left_meters, lidar_bound_left_meters])
        tight_bound_right_meters = np.min([thermal_bound_right_meters, rgb_bound_right_meters, lidar_bound_right_meters])

        return tight_bound_top_meters, tight_bound_bottom_meters, tight_bound_left_meters, tight_bound_right_meters

    def crop_to_tight_bounds(self):
        # tight bounds in pixels
        thermal_tight_bound_top_pixels, thermal_tight_bound_bottom_pixels, thermal_tight_bound_left_pixels, thermal_tight_bound_right_pixels = utils.get_bounds_pixels(bounds_meters=[self.tight_bound_top_meters, self.tight_bound_bottom_meters, self.tight_bound_left_meters, self.tight_bound_right_meters],
                                                                                                                                               origin=[self.thermal_left_meters, self.thermal_top_meters],
                                                                                                                                               res=[self.thermal_res_x, self.thermal_res_y])
        rgb_tight_bound_top_pixels, rgb_tight_bound_bottom_pixels, rgb_tight_bound_left_pixels, rgb_tight_bound_right_pixels = utils.get_bounds_pixels(bounds_meters=[self.tight_bound_top_meters, self.tight_bound_bottom_meters, self.tight_bound_left_meters, self.tight_bound_right_meters],
                                                                                                                                                       origin=[self.rgb_left_meters, self.rgb_top_meters],
                                                                                                                                                       res=[self.rgb_res_x, self.rgb_res_y])
        lidar_tight_bound_top_pixels, lidar_tight_bound_bottom_pixels, lidar_tight_bound_left_pixels, lidar_tight_bound_right_pixels = utils.get_bounds_pixels(bounds_meters=[self.tight_bound_top_meters, self.tight_bound_bottom_meters, self.tight_bound_left_meters, self.tight_bound_right_meters],
                                                                                                                                                               origin=[self.lidar_left_meters, self.lidar_top_meters],
                                                                                                                                                               res=[self.lidar_res_x, self.lidar_res_y])

        # crop to tight bounds
        self.thermal_band = self.thermal_band[thermal_tight_bound_top_pixels : thermal_tight_bound_bottom_pixels, thermal_tight_bound_left_pixels : thermal_tight_bound_right_pixels]
        self.rgb_bands = self.rgb_bands[rgb_tight_bound_top_pixels : rgb_tight_bound_bottom_pixels, rgb_tight_bound_left_pixels : rgb_tight_bound_right_pixels]
        self.lidar_band = self.lidar_band[lidar_tight_bound_top_pixels : lidar_tight_bound_bottom_pixels, lidar_tight_bound_left_pixels : lidar_tight_bound_right_pixels]

    def pad_for_divisibility_by_interval(self):
        self.thermal_band = np.pad(array=self.thermal_band, pad_width=((0, self.thermal_interval - self.thermal_band.shape[0]%self.thermal_interval), (0, self.thermal_interval - self.thermal_band.shape[1]%self.thermal_interval)))
        self.rgb_bands = np.pad(array=self.rgb_bands, pad_width=((0, self.rgb_interval - self.rgb_bands.shape[0]%self.rgb_interval), (0, self.rgb_interval - self.rgb_bands.shape[1]%self.rgb_interval), (0, 0)))
        self.lidar_band = np.pad(array=self.lidar_band, pad_width=((0, self.lidar_interval - self.lidar_band.shape[0]%self.lidar_interval), (0, self.lidar_interval - self.lidar_band.shape[1]%self.lidar_interval)))

    def array_to_tiff(self, array, original_metadata, res, modality):
        metadata = original_metadata.copy()
        metadata['nodata'] = 0
        metadata['width'] = array.shape[1]
        metadata['height'] = array.shape[0]
        metadata['count'] = 1 if len(array.shape) == 2 else 3
        metadata['transform'] = rasterio.transform.Affine(res[0], 0.0, self.tight_bound_left_meters, 0.0, res[1], self.tight_bound_top_meters)

        os.makedirs(f'{self.site_dir}/aligned_tiffs', exist_ok=True)

        with rasterio.open(f'{self.site_dir}/aligned_tiffs/{modality}.tif', 'w', **metadata) as tiff_file:
            if len(array.shape) == 2:
                tiff_file.write(array, 1)
            elif len(array.shape) == 3:
                for i, band in enumerate(array.transpose(2, 0, 1)):
                    tiff_file.write(band, i+1)

if __name__ == '__main__':
    AlignOrthomosaics(site=sys.argv[1])
