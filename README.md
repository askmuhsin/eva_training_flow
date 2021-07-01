# eva_training_flow
This is an updated version of prev training template for eva

to train the model
```python
python main.py
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
    |-- testing.py                    ## test/eval fn
    |-- training.py                   ## training fn
    `-- viz.py                        ## planning to add gradcam, misclassficiation here
```

## TODO
- [ ] add args for epochs, lr `python main.py -lr 0.001 -epochs 100`
- [ ] modify resnet model to have 7X7 channels on layer before GAP
- [ ] remove linear layer, use GAP
- [ ] add gradcam
- [ ] add other visualizations (misclassification code,)
- [ ] add README for traingin flow
- [ ] move regularizations to seperate dir

_Note : the tasks specific to experiment wont be in this repo_
