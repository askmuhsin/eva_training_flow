import torch
import torch.nn.functional as F

def test(
    model, device, 
    test_loader, 
    criterion,
    epoch,
    lr_scheduler=None
):
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            test_loss += criterion(y_pred, target)
            pred = y_pred.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(
        f'TEST \
        Loss:{test_loss:.4f} \
        Acc:{test_acc:.2f} \
        [{correct} / {len(test_loader.dataset)}]'
    )
    
    return test_loss, test_acc
