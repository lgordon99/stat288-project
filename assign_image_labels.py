# imports
import geopandas as gpd
import numpy as np
import pandas as pd
import sys
import utils

class AssignImageLabels:
    def __init__(self, site):
        self.site_dir = utils.get_site_dir(site)
        self.constants = utils.process_yaml('constants.yaml')
        self.classes = ['empty', 'midden', 'mound', 'water']
        self.identifiers = np.load(f'{self.site_dir}/identifiers.npy')
        self.identifier_matrix = np.load(f'{self.site_dir}/identifier_matrix.npy')
        self.labels = np.zeros(len(self.identifiers), dtype=np.int_) 

        self.shapefile_points_to_csv(image_class='mound')
        self.shapefile_points_to_csv(image_class='water')

        for image_class in self.classes[1:]:
            self.assign_class_labels(coordinates_meters=self.get_class_coordinates(image_class), image_class=image_class)

        print('Class breakdown:')
        for image_class in self.classes:
            print(f'{image_class.capitalize()}: {len(self.labels[self.labels == self.classes.index(image_class)])}')
        
        self.label_matrix = np.zeros((self.constants['num_rows_in_tiling'], self.constants['num_cols_in_tiling']), dtype=np.int_)

        for row in range(self.constants['num_rows_in_tiling']):
            for col in range(self.constants['num_cols_in_tiling']):
                self.label_matrix[row][col] = self.labels[np.argmax(self.identifiers == self.identifier_matrix[row][col])]

        np.save(f'{self.site_dir}/labels', self.labels)
        np.save(f'{self.site_dir}/label_matrix', self.label_matrix)

    def shapefile_points_to_csv(self, image_class):
        gdf = gpd.read_file(f'{self.site_dir}/label_data/{image_class}_shapefile/{image_class}_shapefile.shp')
        coordinates_shapefile = np.array([(point.x, point.y) for point in gdf.geometry])

        if coordinates_shapefile[0][1] < 0:
            coordinates_meters = np.zeros((2, len(coordinates_shapefile)))

            for i, [lon, lat] in enumerate(coordinates_shapefile):
                x, y = utils.latlon_to_utm(lat, lon)
                coordinates_meters[0][i] = x
                coordinates_meters[1][i] = y
        else:
            coordinates_meters = coordinates_shapefile.T
        
        df = pd.DataFrame(coordinates_meters.T, columns=['x', 'y'])
        df.to_csv(f'{self.site_dir}/label_data/{image_class}_coordinates_meters.csv', index=False)

    def get_class_coordinates(self, image_class):
        coordinates_meters = pd.read_csv(f'{self.site_dir}/label_data/{image_class}_coordinates_meters.csv').to_numpy().T
        
        return coordinates_meters    

    def assign_class_labels(self, coordinates_meters, image_class):
        coordinates_rows = (coordinates_meters[1] - self.constants['tight_bounds_meters']['top']) / (-self.constants['interval_meters'])
        coordinates_cols = (coordinates_meters[0] - self.constants['tight_bounds_meters']['left']) / self.constants['interval_meters']
        coordinates = np.around([coordinates_rows, coordinates_cols]).T.astype(int)

        assert(len(coordinates_rows[coordinates_rows >= self.identifier_matrix.shape[0]]) == 0)
        assert(len(coordinates_cols[coordinates_cols >= self.identifier_matrix.shape[1]]) == 0)
        print(f'Number of {image_class} instances = {coordinates.shape[0]}')

        for row, col in coordinates:
            if self.identifier_matrix[row][col] in self.identifiers:
                if self.labels[np.argmax(self.identifiers == self.identifier_matrix[row][col])] == 0:
                    self.labels[np.argmax(self.identifiers == self.identifier_matrix[row][col])] = self.classes.index(image_class)
                else:
                    print(f'Identifier {self.identifier_matrix[row][col]} has already been assigned to {self.classes[self.labels[np.argmax(self.identifiers == self.identifier_matrix[row][col])]]}')
            else:
                print('Missing image for identifier')
    
        print(f'{image_class.capitalize()} class labels added')

if __name__ == '__main__':
    AssignImageLabels(site=sys.argv[1])
