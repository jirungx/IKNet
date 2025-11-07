import torch

def predict_model(model, X_test, X_bow, device="cpu"):
    model.eval()
    with torch.no_grad():
        if X_bow is not None:
            pred = model(X_test.to(device), X_bow.to(device))
        else:
            pred = model(X_test.to(device))
    return pred.cpu().numpy()
