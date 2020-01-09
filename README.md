# MFTC
Random drop CNN features

## Usage
### clone
```bash
git clone --recursive https://github.com/BingoH/MFTC.git
```
### run
`random_select.py` options
```bash
python random_select.py -h
```

- pytorch pretrained model
  ```bash
  python random_select.py -a vgg16 -d imagenet -val
  ```

- train and evaluate
  ```bash
  cd pytorch_simple_classification_baselines
  python mnist_train_eval.py

  cd ..
  python random_select.py -a lenet -d mnist --ckp pytorch_simple_classification_baselines/ckpt/lenet_baseline/checkpoint.t7
  ```

## Todos
- remove baseline dep
- to work with py2
