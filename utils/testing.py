import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def test(
    model, device, 
    test_loader, 
    criterion,
    epoch,
    lr_scheduler=None
):
    model.eval()
    pbar = tqdm(test_loader)
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            test_loss += criterion(y_pred, target)
            pred = y_pred.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
            pbar.set_description(
                desc=f'TEST Epoch:{epoch}'
            )

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(
        f'TEST \
        Loss:{test_loss:.4f} \
        Acc:{test_acc:.2f} \
        [{correct} / {len(test_loader.dataset)}]'
    )
    
    return test_loss, test_acc
