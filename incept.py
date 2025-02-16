import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class InceptionKidneyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionKidneyClassifier, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False
            
    def forward(self, x):
        return self.model(x)

def create_datasets(base_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((299, 299)),  # Inception requires 299x299
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    train_dataset = datasets.ImageFolder(
        root=base_dir,
        transform=data_transforms['train'],
        is_valid_file=lambda x: 'train' in x
    )
    
    test_dataset = datasets.ImageFolder(
        root=base_dir,
        transform=data_transforms['test'],
        is_valid_file=lambda x: 'test' in x
    )
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset)
    }
    
    return dataloaders, dataset_sizes

def train_inception(dataloaders, dataset_sizes, device, num_epochs=2):
    model = InceptionKidneyClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    history = {'train_losses': [], 'test_losses': [], 'train_accs': [], 'test_accs': []}
    
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
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Handle auxiliary outputs in training
                    if phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                history['train_losses'].append(epoch_loss)
                history['train_accs'].append(epoch_acc.item())
            else:
                history['test_losses'].append(epoch_loss)
                history['test_accs'].append(epoch_acc.item())
                scheduler.step(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        print()
    
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_curves(history, save_path='inception_training_curves.png'):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.title('Inception - Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accs'], label='Train Accuracy')
    plt.plot(history['test_accs'], label='Test Accuracy')
    plt.title('Inception - Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = "C:\\Users\\gadda\\Downloads\\kidney\\dataset"  # Update this path
    dataloaders, dataset_sizes = create_datasets(data_dir)
    
    model, history = train_inception(dataloaders, dataset_sizes, device)
    torch.save(model.state_dict(), 'inception_kidney_classifier.pth')
    plot_training_curves(history)

if __name__ == "__main__":
    main()

#Best val Acc: 0.954352
# Epoch 1/2
# ----------
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 236/236 [07:28<00:00,  1.90s/it]
# train Loss: 3.6795 Acc: 0.8946
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [01:52<00:00,  1.90s/it]
# test Loss: 0.1592 Acc: 0.9485

# Epoch 2/2
# ----------
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 236/236 [09:16<00:00,  2.36s/it] 
# train Loss: 3.5583 Acc: 0.9450
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 59/59 [02:12<00:00,  2.24s/it] 
# test Loss: 0.1251 Acc: 0.9544