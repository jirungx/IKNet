import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, X_bow, y_train, X_val=None, X_bow_val=None, y_val=None,
                device="cpu", epochs=200, lr=1e-3):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train.to(device), X_bow.to(device)) if X_bow is not None else model(X_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()

        # === 검증 손실 계산 ===
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val.to(device), X_bow_val.to(device)) if X_bow_val is not None else model(X_val.to(device))
                val_loss = criterion(val_out, y_val.to(device)).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = model.state_dict()

            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model
