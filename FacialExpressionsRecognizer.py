import os, os.path
from torchsummary import summary
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image

SEED_VALUE = 42
torch.manual_seed(SEED_VALUE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class RGB2Gray(object):
    def __call__(self, img:Image):
        img = np.asarray(img, dtype=np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        return F.to_pil_image(l_channel)
    
class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_size=8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def __call__(self, img):
        # Convert the PIL image to a numpy array
        img = np.array(img)

        # Apply CLAHE on the single channel image
        clahe = cv2.createCLAHE(self.clip_limit, (self.tile_size, self.tile_size))
        img = clahe.apply(img)

        # Convert the numpy array back to a PIL image
        img = F.to_pil_image(img)

        return img
    
class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self, num_classes, image_channels=1):
        super().__init__()

        # number of channels in input image
        self.image_channels = image_channels
        # number of classes 
        self.num_classes = num_classes
        
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(p=0.10),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=5),

            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout(p=0.10),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.Dropout(p=0.35),

            nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1, padding=0) # output convolutional layer
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        # conv layers
        x = self.conv_layers(x)
        # global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x   

# Training Early Stopper
class TrainingEarlyStopper():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    # to check if training should be stopped
    def should_stop(self, loss):
        should_stop = False
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                should_stop = True
        return should_stop
    
    
# Facial Expressions Image Recognizer
class FacialExpressionsRecognizer:
    def __init__(self,
            image_channels=1, image_width=66, image_height=88,
            train_folder="dataset/train",
            validation_folder="dataset/validation"
        ):
        self.device = "cpu"

        if torch.cuda.is_available():
            self.device = "cuda"
            
        self.classes = list()
        
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height
        self.classes = ImageFolder(self.train_folder).classes
        print("classes= ", self.classes)
        self.num_classes = len(self.classes)

        self.model = ConvolutionalNeuralNetworkModel(self.num_classes, self.image_channels)   
    
    # load pre-trained model
    def load(self, model_file="fcn-facial-expressions.pt"):
        self.model = torch.load(model_file, map_location=self.device)

    # save trained model
    def save(self, model_file):
        torch.save(self.model, model_file)

    def recognize(self, image, seed=SEED_VALUE):
        torch.manual_seed(seed)

        transformations = transforms.Compose([
            RGB2Gray(),
            #CLAHE(clip_limit=1.5, tile_size=8),
            transforms.Resize((self.image_width, self.image_height)),
            transforms.ToTensor(),
        ])

        image = transformations(image)
        # convert to a batch of 1
        input = image.unsqueeze(0).to(self.device)
        # get classifications from model
        output = self.model.to(self.device)(input)
        # get class with highest probability
        _, predicted  = torch.max(output, dim=1)
        # get the class label
        return self.classes[predicted[0].item()]

    def train(self, max_epochs=75, batch_size=64, early_stop:bool=True, opt="SGD", sched="ReduceLROnPlateau", seed=SEED_VALUE):
        torch.manual_seed(seed)

        max_epochs = max_epochs
        batch_size = batch_size
        image_width = self.image_width
        image_height = self.image_height
        num_channels = self.image_channels

        CUDA_AVAILABLE = torch.cuda.is_available()

        if CUDA_AVAILABLE:
            print(f"Using CUDA GPU")
            self.model = self.model.cuda()
       
        print("Model Summary: ")
        print(summary(self.model, (num_channels, image_height, image_width), batch_size))

        transformations = transforms.Compose([
            RGB2Gray(),
            transforms.Resize((self.image_width, self.image_height)),
            #CLAHE(clip_limit=1.5, tile_size=8),
            transforms.RandomHorizontalFlip(p=0.1),  # Randomly flip the image horizontally
            transforms.RandomRotation(3),  # Randomly rotate the image by few degrees
            transforms.ToTensor(),  # Convert the image to a tensor with values between 0 and 1
        ])

        train_dataset = ImageFolder(self.train_folder, transform=transformations)
        val_dataset = ImageFolder(self.validation_folder, transform=transformations)

        num_classes = len(train_dataset.classes)
        self.classes = train_dataset.classes
        print("classes= ", self.classes)
        print("Number of Classes: ", num_classes)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        if opt == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=0.0015)
        elif opt == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=0.0015, momentum=0.9)
        if sched == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, min_lr=1e-7, factor=0.5, verbose=True)
        elif sched == "OnceCycleLR":
            lr_scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=max_epochs*len(train_dataloader), anneal_strategy='cos', pct_start=0.2)
        
        if early_stop:
            early_stopper = TrainingEarlyStopper(patience=15, min_delta=0.00005)
        loss_func = nn.CrossEntropyLoss()

        train_loss = list()
        val_loss = list()
        train_accuracy = list()
        val_accuracy = list()

        print("Training Starting")
        print("-"*100)
        
        writer = SummaryWriter(log_dir="logs/")

        # Training Loop
        for epoch in range(max_epochs):
            # Training
            correct = 0
            iterations = 0
            iteration_loss = 0.0

            self.model.train()

            for i, (inputs, labels) in enumerate(tqdm(train_dataloader, leave=True, desc=f"[Epoch {epoch+1} / {max_epochs}]")):
                if CUDA_AVAILABLE:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = self.model(inputs)
                loss = loss_func(outputs, labels)
                iteration_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1
                
            train_loss.append(iteration_loss / iterations)
            train_accuracy.append(100 * correct / len(train_dataset))

            writer.add_scalar("train_loss", iteration_loss / float(iterations), epoch)
            writer.add_scalar("train_accuracy", (correct / len(train_dataset)), epoch)
            
            # Validation
            loss_validation = 0.0
            correct = 0
            iterations = 0

            self.model.eval()

            for i, (inputs, labels) in enumerate(val_dataloader):
                if CUDA_AVAILABLE:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = self.model(inputs)
                loss = loss_func(outputs, labels)
                loss_validation += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1

            val_loss.append(loss_validation / iterations)
            val_accuracy.append(100 * correct / len(val_dataset))

            # add test loss and accuracy to TensorBoard
            writer.add_scalar("valid_loss", loss_validation / float(iterations), epoch)
            writer.add_scalar("valid_accuracy", (correct / len(val_dataset)), epoch)

            print("Training [ Loss: {:.3f},  Accuracy: {:.3f} ], Validation [ Loss: {:.3f}, Accuracy: {:.3f} ]"
                .format(train_loss[-1], train_accuracy[-1], val_loss[-1], val_accuracy[-1]))

            if sched == "ReduceLROnPlateau":
                lr_scheduler.step(val_loss[-1])
            
            if early_stop:
                if early_stopper.should_stop(val_loss[-1]):
                    break

        print("=== Training Complete ===")


    def evaluate(self, data_dir, device="cuda", show_class_dist=False):
        if torch.cuda.is_available() == False:
            device = "cpu"

        transformations = transforms.Compose([
            RGB2Gray(),
            transforms.Resize((self.image_width, self.image_height)),
            transforms.ToTensor(),
        ])

        dataset = ImageFolder(root=data_dir, transform=transformations)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        classes = dataset.classes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        if show_class_dist:
            # plot class distribution
            labels = [dataset.classes[i] for i in y_true]
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            plt.figure(figsize=(10, 5))
            plt.bar(unique_labels, label_counts)
            plt.xticks(rotation=45)
            plt.xlabel('Class')
            plt.ylabel('Number of Images')
            plt.title('Class Distribution')
            plt.show()