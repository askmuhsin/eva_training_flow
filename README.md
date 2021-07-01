# eva_training_flow
This is an updated version of prev training template for eva

to train the model
```python
python main.py
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
