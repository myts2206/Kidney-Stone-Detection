import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class KidneyClassifier(nn.Module):
    def __init__(self, model_name='vgg16', num_classes=2):
        super(KidneyClassifier, self).__init__()
        
        if model_name == 'vgg16':
            self.base_model = models.vgg16(pretrained=True)
            num_features = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'inception':
            self.base_model = models.inception_v3(pretrained=True)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'mobilenet':
            self.base_model = models.mobilenet_v2(pretrained=True)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier[1] = nn.Linear(num_features, num_classes)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-4]:
            param.requires_grad = False
            
    def forward(self, x):
        return self.base_model(x)

class KidneyDataset:
    def __init__(self, base_dir):
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        # Create datasets for both normal and stone classes
        datasets_train = []
        datasets_test = []
        
        # Path for normal images
        normal_train = os.path.join(base_dir, "normal", "train")
        normal_test = os.path.join(base_dir, "normal", "test")
        
        # Path for stone images
        stone_train = os.path.join(base_dir, "stone", "train")
        stone_test = os.path.join(base_dir, "stone", "test")
        
        # Create train dataset
        if os.path.exists(normal_train) and os.path.exists(stone_train):
            self.train_dataset = datasets.ImageFolder(
                root=base_dir,
                transform=self.data_transforms['train'],
                is_valid_file=lambda x: 'train' in x
            )
        
        # Create test dataset
        if os.path.exists(normal_test) and os.path.exists(stone_test):
            self.test_dataset = datasets.ImageFolder(
                root=base_dir,
                transform=self.data_transforms['test'],
                is_valid_file=lambda x: 'test' in x
            )
        
        # Create dataloaders
        self.dataloaders = {
            'train': DataLoader(
                self.train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=4
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4
            )
        }
        
        self.dataset_sizes = {
            'train': len(self.train_dataset),
            'test': len(self.test_dataset)
        }
        
        self.class_names = ['normal', 'stone']

def train_model(model, dataset, device, num_epochs=2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataset.dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset.dataset_sizes[phase]
            
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                test_losses.append(epoch_loss)
                test_accs.append(epoch_acc.item())
                scheduler.step(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        print()
    
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    
    return model, {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset path - Update this to your path
    data_dir = r"C:\\Users\\gadda\\Downloads\\kidney\\dataset"
    
    # Initialize dataset
    dataset = KidneyDataset(data_dir)
    
    # Train models
    models_to_train = ['vgg16', 'inception', 'mobilenet']
    results = {}
    
    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        model = KidneyClassifier(model_name=model_name)
        model = model.to(device)
        
        # Train the model
        trained_model, history = train_model(model, dataset, device)
        
        # Save the model
        torch.save(trained_model.state_dict(), f'kidney_classifier_{model_name}.pth')
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['test_losses'], label='Test Loss')
        plt.title(f'{model_name} - Training and Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accs'], label='Train Accuracy')
        plt.plot(history['test_accs'], label='Test Accuracy')
        plt.title(f'{model_name} - Training and Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_curves.png')
        plt.close()
        
        results[model_name] = {
            'model': trained_model,
            'history': history
        }
    
    return results

if __name__ == "__main__":
    results = main()

#Best Acc: 0.99542