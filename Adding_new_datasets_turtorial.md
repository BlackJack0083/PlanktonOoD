If you'd like to add your new dataset to this pipeline, this tutorial may help.

## Dataset preparation

### Place your dataset into corresponding position.

Original dataset should be placed in `./data`.

### Creating an imglist for your new dataset

You may use `imglist_generator.py` to generator an imglist for your new dataset, or making an personalize scirpt according to your ideas like `split_dataset_new_class.py`.

> Tip: If your dataset path contains spaces, please note that you need to change `split` to `rsplit` in `line 60` of `openood/datasets/imglist_dataset.py` (we have already made this adjustment).


## Training Stage

Place your `.yml` file into `configs/datasets` as the configuration you like.

## Evaluation Stage

### `eval_ood.py`

You may make any changes in `scripts/eval_ood.py` to fit your new datasets:

#### `Line 57`

```Python
NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200， 'plankton54': 54} # Add your new datasets and the corresponding numbers of classes here.

MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
    'plankton54': ResNet50
    # Add your new datasets and models here.
}
```

### `datasets.py`

You may make any changes in `openood/evaluation_api/datasets.py` to fit your new datasets:

#### `DATA_INFO`

```python 

DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': 'images_classic/',
                'imglist_path': 'benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {...
        },
        'ood': {...
        }
    },
    ...
#Adding your new datasets and configuration here.
}

```

#### `download_id_dict` (optional)

If you'd like to configure an automatically downloadable link for your dataset, you may adding your datasets and links into `download_id_dict` in `openood/evaluation_api/datasets.py`

```python 
download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    ...
    'benchmark_imglist': '1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1'
    # Add your datasets and links here.
}
```


### `preprocessor.py`

You may make any changes in `openood/evaluation_api/preprocessor.py` to fit your new datasets:
