# main.py
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import ImgDataset
from model import get_resnet
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ========== config ==========
DATA_DIR = "./data"
RUNS_DIR = "./runs/Drop_SplitTrain"

NUM_CLASSES = 11
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = correct / total
    return running_loss/total, acc, all_preds, all_labels

def plot_curves(history, save_path):
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc.png'))
    plt.close()

def main():
    # dataset
    train_ds = ImgDataset(DATA_DIR, split="train", val_ratio=0.2)
    val_ds   = ImgDataset(DATA_DIR, split="val", val_ratio=0.2)
    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # model
    model = get_resnet(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # handle class imbalance: compute class weights
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for info in train_ds.data_infos:
        counts[info["label"]] += 1
    print("class counts:", counts)

    #class weights setting
    class_weights = torch.tensor(np.sum(counts) / counts, dtype=torch.float32).to(DEVICE)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    

    best_val_acc = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    for epoch in range(1, 11):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(RUNS_DIR, f"best_resnet50_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'class_to_idx': class_to_idx,
                'epoch': epoch,
                'val_acc': val_acc
            }, save_path)
            print("Saved best model to", save_path)

        # print classification report for val
        print("Validation classification report:")
        print(classification_report(val_labels, val_preds, labels = range(NUM_CLASSES), target_names=[idx_to_class[i] for i in range(NUM_CLASSES)], zero_division=0))

        # plots per few epochs
        if epoch % 5 == 0:
            plot_dir = os.path.join(RUNS_DIR, f"epoch_{epoch}")
            os.makedirs(plot_dir, exist_ok=True)
            plot_curves(history, plot_dir)




    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    for epoch in range(1, EPOCHS+1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(RUNS_DIR, f"best_resnet50_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'class_to_idx': class_to_idx,
                'epoch': epoch,
                'val_acc': val_acc
            }, save_path)
            print("Saved best model to", save_path)

        # print classification report for val
        print("Validation classification report:")
        print(classification_report(val_labels, val_preds, labels = range(NUM_CLASSES), target_names=[idx_to_class[i] for i in range(NUM_CLASSES)], zero_division=0))

        # plots per few epochs
        if epoch % 5 == 0:
            plot_dir = os.path.join(RUNS_DIR, f"epoch_{epoch}")
            os.makedirs(plot_dir, exist_ok=True)
            plot_curves(history, plot_dir)

    # final save of training curves
    plot_curves(history, RUNS_DIR)
    print("Training finished. Best val acc:", best_val_acc)

if __name__ == '__main__':
    main()
