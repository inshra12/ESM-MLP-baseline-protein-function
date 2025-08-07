import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    criterion = nn.BCELoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Run validation after each epoch
        validate_model(model, val_loader, device)

def validate_model(model, val_loader, device='cuda'):
    model.eval()
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    y_pred = torch.cat(all_outputs).numpy()
    y_true = torch.cat(all_targets).numpy()

    y_pred_bin = (y_pred > 0.5).astype(int)  # threshold for multi-label
    f1 = f1_score(y_true, y_pred_bin, average='macro')
    print(f"Validation Macro F1 Score: {f1:.4f}")
