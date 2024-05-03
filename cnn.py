# imports
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torchsummary import summary
from torchvision.models import resnet50
import numpy as np
import random
import torch
import torch.nn as nn
import utils

class CNN:
    def __init__(self, setting, site):
        self.site_dir = utils.get_site_dir(site)
        self.constants = utils.process_yaml('constants.yaml')
        self.classes = ['empty', 'midden', 'mound', 'water']
        self.identifiers = np.load(f'{self.site_dir}/identifiers.npy')
        self.identifier_matrix = np.load(f'{self.site_dir}/identifier_matrix.npy')
        self.labels = np.load(f'{self.site_dir}/labels.npy') 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_batch_size = 32
        self.epochs = 10
        self.criterion = nn.CrossEntropyLoss()

        print(f'Identifiers length = {len(self.identifiers)}')
        print(f'Identifier matrix shape = {self.identifier_matrix.shape}')
        print(f'Labels length = {len(self.labels)}')
        print(f'Class counts = {[len(self.labels[self.labels == self.classes.index(image_class)]) for image_class in self.classes]}')

        train_identifiers = [identifier for identifier in self.identifier_matrix.T[20:].T.ravel() if identifier in self.identifiers] # identifiers in last 61 columns
        train_indices = [np.where(self.identifiers == train_identifier)[0] for train_identifier in train_identifiers] # indices corresponding to those identifiers
        min_nonempty_train_class_count = min([len(np.where(self.labels[train_indices] == self.classes.index(image_class))[0]) for image_class in self.classes[1:]])

        # undersample empty class
        while(len(np.where(self.labels[train_indices] == 0)[0])) > min_nonempty_train_class_count: # more empty images than desired
            del train_indices[random.choice(np.where(self.labels[train_indices] == 0)[0])] # randomly remove one empty image

        train_identifiers = self.identifiers[train_indices]
        test_identifiers = [identifier for identifier in self.identifier_matrix.T[:20].T.ravel() if identifier in self.identifiers]        
        test_indices = [np.where(self.identifiers == test_identifier)[0] for test_identifier in test_identifiers]

        train_labels = self.labels[train_indices].ravel()
        test_labels = self.labels[test_indices].ravel()
        train_class_counts = [len(train_labels[train_labels == self.classes.index(image_class)]) for image_class in self.classes]
        test_class_counts = [len(test_labels[test_labels == self.classes.index(image_class)]) for image_class in self.classes]

        print(f'Train class counts = {train_class_counts}')
        print(f'Test class counts = {test_class_counts}')
        print(f'Train identifiers length = {len(train_identifiers)}')
        print(f'Train indices length = {len(train_indices)}')
        print(f'Test identifiers length = {len(test_identifiers)}')
        print(f'Test indices length = {len(test_indices)}')

        if setting == 'original':
            thermal_tiles = np.load(f'{self.site_dir}/tiles/thermal_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/tiles/rgb_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/tiles/lidar_tiles.npy')

            tripled_thermal_tiles = np.concatenate(([thermal_tiles], [thermal_tiles], [thermal_tiles])).transpose(1, 0, 2, 3)
            tripled_lidar_tiles = np.concatenate(([lidar_tiles], [lidar_tiles], [lidar_tiles])).transpose(1, 0, 2, 3)

            tiles = [tripled_thermal_tiles] + [rgb_tiles.transpose(0, 3, 1, 2)] + [tripled_lidar_tiles]
        elif setting == 'png':
            thermal_tiles = np.load(f'{self.site_dir}/png_tiles/thermal_png_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/png_tiles/rgb_png_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/png_tiles/lidar_png_tiles.npy')

            tiles = np.concatenate(([thermal_tiles], [rgb_tiles], [lidar_tiles])).transpose(0, 1, 4, 2, 3) # (3, 5413, 3, 224, 224)
            print(tiles.shape)
        elif setting == 'fuse':
            fused_tiles = np.load(f'{self.site_dir}/fused_tiles.npy')
            tiles = np.array([fused_tiles.transpose(0, 3, 1, 2)]) # (1, 5413, 3, 224, 224)
        elif setting == 'three_channel':
            thermal_tiles = np.load(f'{self.site_dir}/upsampled_tiles/thermal_upsampled_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/upsampled_tiles/grayscale_upsampled_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/upsampled_tiles/lidar_upsampled_tiles.npy')

            tiles = np.array([np.concatenate(([thermal_tiles], [rgb_tiles], [lidar_tiles])).transpose(1, 0, 2, 3)]) # (1, 5413, 3, 400, 400)
        elif setting == 'five_channel':
            thermal_tiles = np.load(f'{self.site_dir}/upsampled_tiles/thermal_upsampled_tiles.npy')
            rgb_tiles = np.load(f'{self.site_dir}/upsampled_tiles/rgb_upsampled_tiles.npy')
            lidar_tiles = np.load(f'{self.site_dir}/upsampled_tiles/lidar_upsampled_tiles.npy')
            tiles = np.array([np.concatenate(([thermal_tiles], rgb_tiles.transpose(3, 0, 1, 2), [lidar_tiles])).transpose(1, 0, 2, 3)]) # (1, 5413, 5, 400, 400)
            print(tiles.shape)
        
        print(f'Using {torch.cuda.device_count()} GPU(s)')
        model = self.three_channel_model()

        train_images = np.squeeze(np.take(tiles, train_indices, axis=1), axis=2) # (num models, num train images, 3, 224, 224)
        test_images = np.squeeze(np.take(tiles, test_indices, axis=1), axis=2) # (num models, num test images, 3, 224, 224)

        train_means, train_stds = np.zeros((train_images.shape[0], train_images.shape[2])), np.zeros((train_images.shape[0], train_images.shape[2]))

        for i, model_images in enumerate(train_images):
            train_means[i] = np.mean(model_images, axis=(0, 2, 3))
            train_stds[i] = np.std(model_images, axis=(0, 2, 3))

        print(train_means)
        print(train_stds)

        for i in range(len(train_images)):
            train_images[i] = train_images[i] - train_means[i].reshape(1,3,1,1) / train_stds[i].reshape(1,3,1,1)
            test_images[i] = test_images[i] - train_means[i].reshape(1,3,1,1) / train_stds[i].reshape(1,3,1,1)

        print(f'Train images shape = {train_images.shape}')
        print(f'Test images shape = {test_images.shape}')

        train_loader = [self.make_loader(model_images, train_labels, train_identifiers, self.train_batch_size) for model_images in train_images]
        test_loader = [self.make_loader(model_images, test_labels, test_identifiers, self.train_batch_size) for model_images in test_images]

        self.passive_train(model, train_loader[0])
        self.test(model, test_loader[0])

    def get_identifier_label(self, identifier):
        index = np.where(self.identifiers == identifier)[0][0]
        label = self.labels[index]

        return label        

    def three_channel_model(self):
        model = resnet50(weights='DEFAULT') # imports a ResNet-50 CNN pretrained on ImageNet-1k v2

        # for parameter in model.parameters(): # freeze all parameters
        #     parameter.requires_grad = False
        
        num_features = model.fc.in_features # input to final layer
        model.fc = nn.Linear(num_features, len(self.classes)) # unfreezes the parameters in the final layer and sets the output size to the number of classes
        model.to(self.device)
        # print(summary(model, (3, 224, 224)))
        model.eval()
        return model

    def late_fusion_model(self):
        models = [resnet50(weights='DEFAULT') for _ in range(3)]

        for model in models:
            num_features = model.fc.in_features # input to final layer
            model.fc = nn.Linear(num_features, 256)
        
        fusion_fc = nn.Linear(3*256, len(self.classes))

        for model in models:
            model.to(self.device)

    def make_loader(self, images, labels, identifiers, batch_size):
        data = list(map(list, zip(images, labels, identifiers))) # each image gets grouped with its label and identifier
        data = random.sample(data, len(data)) # shuffle the training data
        loader = {}
        image_batch = []
        label_batch = []
        identifier_batch = []
        batch_number = 0

        # batch the data
        for i in range(len(data) + 1): 
            if (i % batch_size == 0 and i != 0) or (i == len(data)):
                loader[f'batch {batch_number}'] = {}
                loader[f'batch {batch_number}']['images'] = torch.tensor(np.array(image_batch)).float()
                loader[f'batch {batch_number}']['labels'] = torch.tensor(np.array(label_batch))
                loader[f'batch {batch_number}']['identifiers'] = identifier_batch
                image_batch = []
                label_batch = []
                identifier_batch = []
                batch_number += 1

            if i != len(data):
                image_batch.append(data[i][0])
                label_batch.append(data[i][1])
                identifier_batch.append(data[i][2])
        
        return loader

    def passive_train(self, model, train_loader):
        model.train()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(self.epochs):
            epoch_loss = 0

            for batch in train_loader:
                images = train_loader[batch]['images'].to(self.device)
                labels = train_loader[batch]['labels'].to(self.device)
                optimizer.zero_grad() # zero the parameter gradients
                outputs = model(images) # forward pass
                loss = self.criterion(outputs, labels) # mean loss per item in batch
                loss.backward() # backward pass
                optimizer.step() # optimization
                epoch_loss += loss.item()

            print(f'Epoch {epoch+1} loss = {round(epoch_loss, 3)}')

        model.eval()

    def test(self, model, test_loader):
        y_true = torch.tensor([], dtype=torch.long, device=self.device)
        probabilities = torch.tensor([], dtype=torch.long, device=self.device)

        with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs
            for batch in test_loader:
                images = test_loader[batch]['images'].to(self.device)
                labels = test_loader[batch]['labels'].to(self.device)
                outputs = model(images) # forward pass
                batch_probabilities = torch.nn.functional.softmax(outputs, dim=1) # applies softmax to the logits

                y_true = torch.cat((y_true, labels), dim=0)
                probabilities = torch.cat((probabilities, batch_probabilities), dim=0)

        y_pred = torch.argmax(probabilities, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        print(y_true.shape)
        print(probabilities.shape)
        print(y_pred.shape)

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, probabilities, multi_class='ovr')

        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-Score:', f1)
        print('ROC AUC:', roc_auc)
        # print('Cross-Entropy Loss:', cross_entropy_loss)

if __name__ == '__main__':
    # CNN(setting='fuse', site='firestorm-3')
    CNN(setting='three_channel', site='firestorm-3')
    # CNN(setting='png', site='firestorm-3')
    # CNN(setting='original', site='firestorm-3')
    # CNN(setting='five_channel', site='firestorm-3')
