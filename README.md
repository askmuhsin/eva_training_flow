# eva_training_flow

To train the model use the Trainer class 👇
```python
from models import resnet_v2_6ch_ending     ## import model
from main import Trainer
from main import show_misclassification     ## utility to show misclassifications and gradcam

trainer = Trainer(
    resnet_v2_6ch_ending.ResNet18(),
    # model_path='../data/model_state/R18_6_channel_with_augmentation_3_repeat.pt',     ## model path is optional, if required to resume training
)
trainer.train_model(epochs=40)  ## to start training
show_misclassification(trainer) ## to view misclassifications
```

## Flow Structure
```
.
|-- main.py
|-- models                             ## directory for model files
|   |-- __init__.py
|   `-- resnet.py
|-- README.md
|-- tests                             ## add tests, TODO
|   `-- test_modules_v1.ipynb
`-- utils                             ## all utils go here
    |-- __init__.py
    |-- data.py                       ## DataLoader
    |-- regularizations.py            ## Regularizations, album lib
    |-- setup.py                      ## device, cuda, ...
    |-- testing.py                    ## test fns, 
    |-- training.py                   ## training fn
    `-- viz.py                        ## gradcam, and probability visualizations
```

