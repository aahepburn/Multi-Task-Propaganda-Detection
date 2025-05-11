import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm



def predict_proba(model, loader, task, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, task=task)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.extend(probs)

    return np.array(preds)


# ==========================
# DATASET CLASS
# ==========================

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }



# ==========================
# MODEL CLASS
# ==========================

class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)





def train_single_task_model(model,
                            train_loader,
                            val_loader,
                            y_val,
                            MODEL_PATH,
                            LEARNING_RATE=2e-5,
                            EPOCHS=3,
                            device=None,
                            predict_proba=None,
                            evaluate_threshold_sweep=None):
    """
    Trains a single-task multi-label classifier using BCEWithLogitsLoss.
    Preserves detailed logging and saves best model based on macro F1.

    Args:
        model (nn.Module): Transformer-based classifier.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        y_val (np.array): Binary matrix of true labels for validation.
        MODEL_PATH (str): File path to save the best model.
        LEARNING_RATE (float): Learning rate for optimizer.
        EPOCHS (int): Number of training epochs.
        device (torch.device): CUDA or CPU device.
        predict_proba (function): Function to generate probabilities from model.
        evaluate_threshold_sweep (function): Function to find best threshold.

    Returns:
        nn.Module: Trained model (best epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_macro_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")

        # Validation
        val_probs = predict_proba(model, val_loader, device)
        threshold = evaluate_threshold_sweep(y_val, val_probs)
        y_val_pred = (val_probs > threshold).astype(int)
        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        print(f"Validation Macro F1 (Epoch {epoch + 1}): {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (Epoch {epoch + 1}) to {MODEL_PATH}")

    return model

'''
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

def train_single_task_model(model,
                            train_loader,
                            val_loader,
                            y_val,
                            MODEL_PATH,
                            LEARNING_RATE=2e-5,
                            EPOCHS=3,
                            device=None):
    """
    Trains a single-task multi-label classifier using BCEWithLogitsLoss.
    Preserves detailed logging and saves best model based on macro F1.

    Args:
        model (nn.Module): Transformer-based classifier.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        y_val (np.array): Binary matrix of true labels for validation.
        MODEL_PATH (str): File path to save the best model.
        LEARNING_RATE (float): Learning rate for optimizer.
        EPOCHS (int): Number of training epochs.
        device (torch.device): CUDA or CPU device.

    Returns:
        nn.Module: Trained model (best epoch).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_macro_f1 = 0.0
    best_state_dict = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].cpu().numpy()

                outputs = model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels)

        y_true = np.array(all_labels)
        y_pred_bin = (np.array(all_probs) > 0.35).astype(int)  # ← Fixed threshold
        macro_f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
        print(f"Validation Macro F1 (Epoch {epoch + 1}): {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, MODEL_PATH)
            print(f"Saved best model (Epoch {epoch + 1}) to {MODEL_PATH}")

    if best_state_dict:
        model.load_state_dict(best_state_dict)

    return model
'''

def hierarchical_loss(logits, targets, child_to_parent, label_to_index):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    base_loss = bce(logits, targets)
    for child, parent in child_to_parent.items():
        if child in label_to_index and parent in label_to_index:
            child_idx = label_to_index[child]
            parent_idx = label_to_index[parent]
            mask = targets[:, parent_idx]
            base_loss[:, child_idx] *= mask
    return base_loss.mean()

def apply_hierarchical_constraints(preds, child_to_parent, label_to_index):
    for child, parent in child_to_parent.items():
        if child in label_to_index and parent in label_to_index:
            child_idx = label_to_index[child]
            parent_idx = label_to_index[parent]
            preds[:, child_idx] *= preds[:, parent_idx] >= 0.5
    return preds



from sklearn.metrics import f1_score
from tqdm import tqdm

def train_hierarchical_classifier(model,train_loader, val_loader,
                                  y_val, MODEL_PATH,
                                  child_to_parent, label_to_index,
                                  predict_proba, evaluate_threshold_sweep,
                                  LEARNING_RATE=2e-5, EPOCHS=3):
    """
    Trains a transformer-based hierarchical classifier using hierarchical loss and constraints.

    Args:
        model (nn.Module): The transformer-based classifier model.
        train_loader (DataLoader): Training set loader.
        val_loader (DataLoader): Validation set loader.
        y_val (np.array): Ground-truth binary labels for validation set.
        MODEL_PATH (str): Where to save the best model.
        child_to_parent (dict): Mapping from fine to coarse labels.
        label_to_index (dict): Mapping from label name to column index.
        predict_proba (function): Probabilistic prediction function.
        evaluate_threshold_sweep (function): Threshold optimizer.
        LEARNING_RATE (float): Learning rate.
        EPOCHS (int): Number of epochs.

    Returns:
        model (nn.Module): Trained model (best checkpoint).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_macro_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = hierarchical_loss(outputs, labels, child_to_parent, label_to_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

        val_probs = predict_proba(model, val_loader, device)
        threshold = evaluate_threshold_sweep(y_val, val_probs)
        y_val_pred_tensor = torch.tensor((val_probs > threshold).astype(float)).float()
        y_val_pred_tensor = apply_hierarchical_constraints(y_val_pred_tensor, child_to_parent, label_to_index)
        y_val_pred = y_val_pred_tensor.cpu().numpy().astype(int)

        macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        print(f"Validation Macro F1 (Epoch {epoch+1}): {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model (Epoch {epoch+1}) to {MODEL_PATH}")

    return model



