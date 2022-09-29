## Setup environment Verify Monai running on exascale.mahidol.ac.th

#### With terminal at any Compute Node, run Docker image with Singularity
$ singularity run --nv docker://projectmonai/monai

#### Run notebook
$ singularity shell --nv docker://projectmonai/monai

Singularity> jupyter lab


# Medical Image Classification Tutorial with the MedNIST Dataset

In this tutorial, we introduce an end-to-end training and evaluation example based on the MedNIST dataset.

We'll go through the following steps:
* Create a dataset for training and testing
* Use MONAI transforms to pre-process data
* Use the DenseNet from MONAI for classification
* Train the model with a PyTorch program
* Evaluate on test dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb)


```python
import torch
```


```python
!python -c "import monai" || pip install -q "monai-weekly[pillow, tqdm]"
!python -c "import matplotlib" || pip install -q matplotlib
%matplotlib inline
```


```python
!nvidia-smi
```

    Fri Sep 23 17:26:58 2022       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.7     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
    | N/A   34C    P0    68W / 400W |   3802MiB / 81251MiB |      5%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   1  NVIDIA A100-SXM...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   28C    P0    59W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   2  NVIDIA A100-SXM...  On   | 00000000:47:00.0 Off |                    0 |
    | N/A   29C    P0    58W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   3  NVIDIA A100-SXM...  On   | 00000000:4E:00.0 Off |                    0 |
    | N/A   30C    P0    58W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   4  NVIDIA A100-SXM...  On   | 00000000:87:00.0 Off |                    0 |
    | N/A   37C    P0    58W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   5  NVIDIA A100-SXM...  On   | 00000000:90:00.0 Off |                    0 |
    | N/A   33C    P0    64W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   6  NVIDIA A100-SXM...  On   | 00000000:B7:00.0 Off |                    0 |
    | N/A   33C    P0    61W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
    |   7  NVIDIA A100-SXM...  On   | 00000000:BD:00.0 Off |                    0 |
    | N/A   33C    P0    59W / 400W |      3MiB / 81251MiB |      0%      Default |
    |                               |                      |             Disabled |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A   2429330      C   /opt/conda/bin/python3.8         3799MiB |
    +-----------------------------------------------------------------------------+


## Setup imports


```python
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

print_config()
```

    MONAI version: 1.0.0+5.g84e271ec
    Numpy version: 1.22.4
    Pytorch version: 1.13.0a0+d321be6
    MONAI flags: HAS_EXT = True, USE_COMPILED = False, USE_META_DICT = False
    MONAI rev id: 84e271ec939330e7cedf22b3871c4a2a62d3c2a2
    MONAI __file__: /opt/monai/monai/__init__.py
    
    Optional dependencies:
    Pytorch Ignite version: 0.4.10
    Nibabel version: 4.0.2
    scikit-image version: 0.19.3
    Pillow version: 9.0.1
    Tensorboard version: 2.9.1
    gdown version: 4.5.1
    TorchVision version: 0.14.0a0
    tqdm version: 4.62.3
    lmdb version: 1.3.0
    psutil version: 5.9.0
    pandas version: 1.3.5
    einops version: 0.4.1
    transformers version: 4.21.3
    mlflow version: 1.29.0
    pynrrd version: 0.4.3
    
    For details about installing the optional dependencies, please visit:
        https://docs.monai.io/en/latest/installation.html#installing-the-recommended-dependencies
    


## Setup data directory

You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.  
This allows you to save results and reuse downloads.  
If not specified a temporary directory will be used.


```python
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
```

    /tmp/tmpbasuc2tx


## Download dataset

The MedNIST dataset was gathered from several sets from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions),
[the RSNA Bone Age Challenge](http://rsnachallenges.cloudapp.net/competitions/4),
and [the NIH Chest X-ray dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest).

The dataset is kindly made available by [Dr. Bradley J. Erickson M.D., Ph.D.](https://www.mayo.edu/research/labs/radiology-informatics/overview) (Department of Radiology, Mayo Clinic)
under the Creative Commons [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

If you use the MedNIST dataset, please acknowledge the source.


```python
resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
md5 = "0bc7306e7427e00ad1c5526a6677552d"

compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
data_dir = os.path.join(root_dir, "MedNIST")
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, root_dir, md5)
```

    MedNIST.tar.gz: 59.0MB [00:07, 8.56MB/s]                                                               

    2022-09-23 17:24:15,104 - INFO - Downloaded: /tmp/tmpbasuc2tx/MedNIST.tar.gz
    2022-09-23 17:24:15,204 - INFO - Verified 'MedNIST.tar.gz', md5: 0bc7306e7427e00ad1c5526a6677552d.
    2022-09-23 17:24:15,205 - INFO - Writing into directory: /tmp/tmpbasuc2tx.


    


## Set deterministic training for reproducibility


```python
set_determinism(seed=0)
```

## Read image filenames from the dataset folders

First of all, check the dataset files and show some statistics.  
There are 6 folders in the dataset: Hand, AbdomenCT, CXR, ChestCT, BreastMRI, HeadCT,  
which should be used as the labels to train our classification model.


```python
class_names = sorted(x for x in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
image_files = [
    [
        os.path.join(data_dir, class_names[i], x)
        for x in os.listdir(os.path.join(data_dir, class_names[i]))
    ]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")
```

    Total image count: 58954
    Image dimensions: 64 x 64
    Label names: ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
    Label counts: [10000, 8954, 10000, 10000, 10000, 10000]


## Randomly pick images from the dataset to visualize and check


```python
plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = PIL.Image.open(image_files_list[k])
    arr = np.array(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
plt.tight_layout()
plt.show()
```


    
![Jupyter Notebook Plot](/assets/notebooks/2022-09-23-Monai_mednist_tutorial_files/2022-09-23-Monai_mednist_tutorial_16_0.png)
    


## Prepare training, validation and test data lists

Randomly select 10% of the dataset as validation and 10% as test.


```python
val_frac = 0.1
test_frac = 0.1
length = len(image_files_list)
indices = np.arange(length)
np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
val_indices = indices[test_split:val_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list[i] for i in val_indices]
val_y = [image_class[i] for i in val_indices]
test_x = [image_files_list[i] for i in test_indices]
test_y = [image_class[i] for i in test_indices]

print(
    f"Training count: {len(train_x)}, Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")
```

    Training count: 47164, Validation count: 5895, Test count: 5895


## Define MONAI transforms, Dataset and Dataloader to pre-process data


```python
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])
```


```python
class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = MedNISTDataset(train_x, train_y, train_transforms)
train_loader = DataLoader(
    train_ds, batch_size=300, shuffle=True, num_workers=10)

val_ds = MedNISTDataset(val_x, val_y, val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=300, num_workers=10)

test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_loader = DataLoader(
    test_ds, batch_size=300, num_workers=10)
```

## Define network and optimizer

1. Set learning rate for how much the model is updated per batch.
1. Set total epoch number, as we have shuffle and random transforms, so the training data of every epoch is different.  
And as this is just a get start tutorial, let's just train 4 epochs.  
If train 10 epochs, the model can achieve 100% accuracy on test dataset. 
1. Use DenseNet from MONAI and move to GPU devide, this DenseNet can support both 2D and 3D classification tasks.
1. Use Adam optimizer.


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNet121(spatial_dims=2, in_channels=1,
                    out_channels=num_class).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 4
val_interval = 1
auc_metric = ROCAUCMetric()
```


```python
print(device)
```

    cuda


## Model training

Execute a typical PyTorch training that run epoch loop and step loop, and do validation after every epoch.  
Will save the model weights to file if got best validation accuracy.


```python
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")
```

    ----------
    epoch 1/4
    1/157, train_loss: 1.7905
    2/157, train_loss: 1.7626
    3/157, train_loss: 1.7371
    4/157, train_loss: 1.7116
    5/157, train_loss: 1.6850
    6/157, train_loss: 1.6477
    7/157, train_loss: 1.6469
    8/157, train_loss: 1.5967
    9/157, train_loss: 1.5772
    10/157, train_loss: 1.5541
    11/157, train_loss: 1.5259
    12/157, train_loss: 1.5018
    13/157, train_loss: 1.4797
    14/157, train_loss: 1.4746
    15/157, train_loss: 1.4627
    16/157, train_loss: 1.4242
    17/157, train_loss: 1.4104
    18/157, train_loss: 1.3556
    19/157, train_loss: 1.3459
    20/157, train_loss: 1.3497
    21/157, train_loss: 1.3167
    22/157, train_loss: 1.3070
    23/157, train_loss: 1.2892
    24/157, train_loss: 1.2665
    25/157, train_loss: 1.2887
    26/157, train_loss: 1.2419
    27/157, train_loss: 1.2158
    28/157, train_loss: 1.2117
    29/157, train_loss: 1.1599
    30/157, train_loss: 1.1648
    31/157, train_loss: 1.1376
    32/157, train_loss: 1.1325
    33/157, train_loss: 1.1057
    34/157, train_loss: 1.0882
    35/157, train_loss: 1.0878
    36/157, train_loss: 1.0690
    37/157, train_loss: 1.0404
    38/157, train_loss: 1.0373
    39/157, train_loss: 1.0288
    40/157, train_loss: 1.0198
    41/157, train_loss: 1.0534
    42/157, train_loss: 0.9989
    43/157, train_loss: 0.9581
    44/157, train_loss: 0.9800
    45/157, train_loss: 0.9438
    46/157, train_loss: 0.9364
    47/157, train_loss: 0.9325
    48/157, train_loss: 0.8583
    49/157, train_loss: 0.9325
    50/157, train_loss: 0.8685
    51/157, train_loss: 0.8607
    52/157, train_loss: 0.8824
    53/157, train_loss: 0.8850
    54/157, train_loss: 0.8236
    55/157, train_loss: 0.8076
    56/157, train_loss: 0.8243
    57/157, train_loss: 0.8005
    58/157, train_loss: 0.8184
    59/157, train_loss: 0.8012
    60/157, train_loss: 0.7567
    61/157, train_loss: 0.7632
    62/157, train_loss: 0.7396
    63/157, train_loss: 0.7593
    64/157, train_loss: 0.7228
    65/157, train_loss: 0.7203
    66/157, train_loss: 0.7411
    67/157, train_loss: 0.7002
    68/157, train_loss: 0.7009
    69/157, train_loss: 0.7318
    70/157, train_loss: 0.6492
    71/157, train_loss: 0.6831
    72/157, train_loss: 0.6540
    73/157, train_loss: 0.6611
    74/157, train_loss: 0.6371
    75/157, train_loss: 0.6515
    76/157, train_loss: 0.6436
    77/157, train_loss: 0.6124
    78/157, train_loss: 0.5939
    79/157, train_loss: 0.6436
    80/157, train_loss: 0.6096
    81/157, train_loss: 0.5772
    82/157, train_loss: 0.6238
    83/157, train_loss: 0.5836
    84/157, train_loss: 0.5476
    85/157, train_loss: 0.5270
    86/157, train_loss: 0.5493
    87/157, train_loss: 0.5139
    88/157, train_loss: 0.5316
    89/157, train_loss: 0.4883
    90/157, train_loss: 0.5082
    91/157, train_loss: 0.5078
    92/157, train_loss: 0.5019
    93/157, train_loss: 0.4842
    94/157, train_loss: 0.4924
    95/157, train_loss: 0.4652
    96/157, train_loss: 0.4531
    97/157, train_loss: 0.4363
    98/157, train_loss: 0.4830
    99/157, train_loss: 0.4879
    100/157, train_loss: 0.4651
    101/157, train_loss: 0.4365
    102/157, train_loss: 0.4504
    103/157, train_loss: 0.4245
    104/157, train_loss: 0.4201
    105/157, train_loss: 0.4420
    106/157, train_loss: 0.4181
    107/157, train_loss: 0.4398
    108/157, train_loss: 0.4444
    109/157, train_loss: 0.4060
    110/157, train_loss: 0.4293
    111/157, train_loss: 0.3760
    112/157, train_loss: 0.3841
    113/157, train_loss: 0.3836
    114/157, train_loss: 0.3843
    115/157, train_loss: 0.3926
    116/157, train_loss: 0.3797
    117/157, train_loss: 0.3463
    118/157, train_loss: 0.3594
    119/157, train_loss: 0.3682
    120/157, train_loss: 0.3729
    121/157, train_loss: 0.3252
    122/157, train_loss: 0.3360
    123/157, train_loss: 0.3300
    124/157, train_loss: 0.3278
    125/157, train_loss: 0.3313
    126/157, train_loss: 0.3747
    127/157, train_loss: 0.3247
    128/157, train_loss: 0.3116
    129/157, train_loss: 0.3438
    130/157, train_loss: 0.2886
    131/157, train_loss: 0.3485
    132/157, train_loss: 0.3560
    133/157, train_loss: 0.3011
    134/157, train_loss: 0.3300
    135/157, train_loss: 0.3039
    136/157, train_loss: 0.2945
    137/157, train_loss: 0.3143
    138/157, train_loss: 0.2705
    139/157, train_loss: 0.3192
    140/157, train_loss: 0.3055
    141/157, train_loss: 0.2892
    142/157, train_loss: 0.2747
    143/157, train_loss: 0.2495
    144/157, train_loss: 0.2922
    145/157, train_loss: 0.2819
    146/157, train_loss: 0.2984
    147/157, train_loss: 0.2438
    148/157, train_loss: 0.2666
    149/157, train_loss: 0.2859
    150/157, train_loss: 0.2713
    151/157, train_loss: 0.2337
    152/157, train_loss: 0.2684
    153/157, train_loss: 0.2396
    154/157, train_loss: 0.2412
    155/157, train_loss: 0.2665
    156/157, train_loss: 0.2361
    157/157, train_loss: 0.2319
    158/157, train_loss: 0.2651
    epoch 1 average loss: 0.7282
    saved new best metric model
    current epoch: 1 current AUC: 0.9979 current accuracy: 0.9628 best AUC: 0.9979 at epoch: 1
    ----------
    epoch 2/4
    1/157, train_loss: 0.2430
    2/157, train_loss: 0.2423
    3/157, train_loss: 0.2449
    4/157, train_loss: 0.2248
    5/157, train_loss: 0.2292
    6/157, train_loss: 0.2180
    7/157, train_loss: 0.2514
    8/157, train_loss: 0.2432
    9/157, train_loss: 0.2361
    10/157, train_loss: 0.1946
    11/157, train_loss: 0.1935
    12/157, train_loss: 0.2468
    13/157, train_loss: 0.2618
    14/157, train_loss: 0.2052
    15/157, train_loss: 0.2026
    16/157, train_loss: 0.2044
    17/157, train_loss: 0.2144
    18/157, train_loss: 0.2303
    19/157, train_loss: 0.2084
    20/157, train_loss: 0.2019
    21/157, train_loss: 0.1884
    22/157, train_loss: 0.2108
    23/157, train_loss: 0.2110
    24/157, train_loss: 0.1985
    25/157, train_loss: 0.2192
    26/157, train_loss: 0.1841
    27/157, train_loss: 0.2212
    28/157, train_loss: 0.1863
    29/157, train_loss: 0.1887
    30/157, train_loss: 0.1960
    31/157, train_loss: 0.1713
    32/157, train_loss: 0.1688
    33/157, train_loss: 0.1674
    34/157, train_loss: 0.1880
    35/157, train_loss: 0.1666
    36/157, train_loss: 0.1842
    37/157, train_loss: 0.1713
    38/157, train_loss: 0.1974
    39/157, train_loss: 0.1810
    40/157, train_loss: 0.1947
    41/157, train_loss: 0.1749
    42/157, train_loss: 0.2326
    43/157, train_loss: 0.2106
    44/157, train_loss: 0.1624
    45/157, train_loss: 0.1487
    46/157, train_loss: 0.1724
    47/157, train_loss: 0.1669
    48/157, train_loss: 0.1736
    49/157, train_loss: 0.1775
    50/157, train_loss: 0.1723
    51/157, train_loss: 0.1735
    52/157, train_loss: 0.1559
    53/157, train_loss: 0.1526
    54/157, train_loss: 0.1601
    55/157, train_loss: 0.1570
    56/157, train_loss: 0.1619
    57/157, train_loss: 0.1541
    58/157, train_loss: 0.1830
    59/157, train_loss: 0.1526
    60/157, train_loss: 0.1895
    61/157, train_loss: 0.1593
    62/157, train_loss: 0.1595
    63/157, train_loss: 0.1466
    64/157, train_loss: 0.1269
    65/157, train_loss: 0.1215
    66/157, train_loss: 0.1434
    67/157, train_loss: 0.1469
    68/157, train_loss: 0.1449
    69/157, train_loss: 0.1234
    70/157, train_loss: 0.1674
    71/157, train_loss: 0.1498
    72/157, train_loss: 0.1493
    73/157, train_loss: 0.1147
    74/157, train_loss: 0.1362
    75/157, train_loss: 0.1311
    76/157, train_loss: 0.1126
    77/157, train_loss: 0.1231
    78/157, train_loss: 0.1362
    79/157, train_loss: 0.1214
    80/157, train_loss: 0.1322
    81/157, train_loss: 0.1094
    82/157, train_loss: 0.1324
    83/157, train_loss: 0.1329
    84/157, train_loss: 0.1064
    85/157, train_loss: 0.1402
    86/157, train_loss: 0.1354
    87/157, train_loss: 0.1250
    88/157, train_loss: 0.1310
    89/157, train_loss: 0.1087
    90/157, train_loss: 0.1167
    91/157, train_loss: 0.1204
    92/157, train_loss: 0.1244
    93/157, train_loss: 0.1250
    94/157, train_loss: 0.1424
    95/157, train_loss: 0.1328
    96/157, train_loss: 0.1088
    97/157, train_loss: 0.1317
    98/157, train_loss: 0.0987
    99/157, train_loss: 0.1127
    100/157, train_loss: 0.0909
    101/157, train_loss: 0.1434
    102/157, train_loss: 0.1138
    103/157, train_loss: 0.1210
    104/157, train_loss: 0.0901
    105/157, train_loss: 0.0986
    106/157, train_loss: 0.1226
    107/157, train_loss: 0.1076
    108/157, train_loss: 0.1164
    109/157, train_loss: 0.1077
    110/157, train_loss: 0.1028
    111/157, train_loss: 0.0874
    112/157, train_loss: 0.0962
    113/157, train_loss: 0.1147
    114/157, train_loss: 0.0992
    115/157, train_loss: 0.0848
    116/157, train_loss: 0.1218
    117/157, train_loss: 0.0939
    118/157, train_loss: 0.1227
    119/157, train_loss: 0.1069
    120/157, train_loss: 0.1095
    121/157, train_loss: 0.1252
    122/157, train_loss: 0.0996
    123/157, train_loss: 0.0844
    124/157, train_loss: 0.0979
    125/157, train_loss: 0.1441
    126/157, train_loss: 0.1036
    127/157, train_loss: 0.1001
    128/157, train_loss: 0.0950
    129/157, train_loss: 0.1022
    130/157, train_loss: 0.0776
    131/157, train_loss: 0.0850
    132/157, train_loss: 0.1019
    133/157, train_loss: 0.1034
    134/157, train_loss: 0.0910
    135/157, train_loss: 0.0986
    136/157, train_loss: 0.0765
    137/157, train_loss: 0.0908
    138/157, train_loss: 0.1176
    139/157, train_loss: 0.1113
    140/157, train_loss: 0.0779
    141/157, train_loss: 0.0871
    142/157, train_loss: 0.0958
    143/157, train_loss: 0.0876
    144/157, train_loss: 0.1181
    145/157, train_loss: 0.1112
    146/157, train_loss: 0.0980
    147/157, train_loss: 0.0933
    148/157, train_loss: 0.1106
    149/157, train_loss: 0.0818
    150/157, train_loss: 0.0976
    151/157, train_loss: 0.1008
    152/157, train_loss: 0.0950
    153/157, train_loss: 0.0954
    154/157, train_loss: 0.0822
    155/157, train_loss: 0.0936
    156/157, train_loss: 0.0946
    157/157, train_loss: 0.0893
    158/157, train_loss: 0.2379
    epoch 2 average loss: 0.1450
    saved new best metric model
    current epoch: 2 current AUC: 0.9998 current accuracy: 0.9869 best AUC: 0.9998 at epoch: 2
    ----------
    epoch 3/4
    1/157, train_loss: 0.0969
    2/157, train_loss: 0.0851
    3/157, train_loss: 0.0765
    4/157, train_loss: 0.0736
    5/157, train_loss: 0.0909
    6/157, train_loss: 0.0788
    7/157, train_loss: 0.0887
    8/157, train_loss: 0.0749
    9/157, train_loss: 0.0842
    10/157, train_loss: 0.0837
    11/157, train_loss: 0.0768
    12/157, train_loss: 0.0731
    13/157, train_loss: 0.0827
    14/157, train_loss: 0.0692
    15/157, train_loss: 0.0691
    16/157, train_loss: 0.0903
    17/157, train_loss: 0.0774
    18/157, train_loss: 0.0843
    19/157, train_loss: 0.0715
    20/157, train_loss: 0.0758
    21/157, train_loss: 0.0868
    22/157, train_loss: 0.0696
    23/157, train_loss: 0.0775
    24/157, train_loss: 0.1073
    25/157, train_loss: 0.1210
    26/157, train_loss: 0.0668
    27/157, train_loss: 0.0599
    28/157, train_loss: 0.0633
    29/157, train_loss: 0.0760
    30/157, train_loss: 0.0899
    31/157, train_loss: 0.0845
    32/157, train_loss: 0.0889
    33/157, train_loss: 0.0737
    34/157, train_loss: 0.0703
    35/157, train_loss: 0.0758
    36/157, train_loss: 0.0711
    37/157, train_loss: 0.0781
    38/157, train_loss: 0.0633
    39/157, train_loss: 0.0774
    40/157, train_loss: 0.0615
    41/157, train_loss: 0.0657
    42/157, train_loss: 0.0891
    43/157, train_loss: 0.0927
    44/157, train_loss: 0.0674
    45/157, train_loss: 0.0734
    46/157, train_loss: 0.0864
    47/157, train_loss: 0.0567
    48/157, train_loss: 0.0922
    49/157, train_loss: 0.0613
    50/157, train_loss: 0.0696
    51/157, train_loss: 0.0959
    52/157, train_loss: 0.0852
    53/157, train_loss: 0.0843
    54/157, train_loss: 0.0643
    55/157, train_loss: 0.0666
    56/157, train_loss: 0.0948
    57/157, train_loss: 0.0620
    58/157, train_loss: 0.0641
    59/157, train_loss: 0.0878
    60/157, train_loss: 0.0607
    61/157, train_loss: 0.0649
    62/157, train_loss: 0.0662
    63/157, train_loss: 0.0575
    64/157, train_loss: 0.0591
    65/157, train_loss: 0.0697
    66/157, train_loss: 0.0640
    67/157, train_loss: 0.0678
    68/157, train_loss: 0.0680
    69/157, train_loss: 0.0685
    70/157, train_loss: 0.0630
    71/157, train_loss: 0.0750
    72/157, train_loss: 0.0575
    73/157, train_loss: 0.0703
    74/157, train_loss: 0.0469
    75/157, train_loss: 0.0486
    76/157, train_loss: 0.0643
    77/157, train_loss: 0.0665
    78/157, train_loss: 0.0681
    79/157, train_loss: 0.0411
    80/157, train_loss: 0.0639
    81/157, train_loss: 0.0644
    82/157, train_loss: 0.0619
    83/157, train_loss: 0.0713
    84/157, train_loss: 0.0468
    85/157, train_loss: 0.0822
    86/157, train_loss: 0.0543
    87/157, train_loss: 0.0633
    88/157, train_loss: 0.0614
    89/157, train_loss: 0.0561
    90/157, train_loss: 0.0612
    91/157, train_loss: 0.0459
    92/157, train_loss: 0.0551
    93/157, train_loss: 0.0573
    94/157, train_loss: 0.0616
    95/157, train_loss: 0.0581
    96/157, train_loss: 0.0576
    97/157, train_loss: 0.0708
    98/157, train_loss: 0.0520
    99/157, train_loss: 0.0504
    100/157, train_loss: 0.0614
    101/157, train_loss: 0.0548
    102/157, train_loss: 0.0600
    103/157, train_loss: 0.0431
    104/157, train_loss: 0.0687
    105/157, train_loss: 0.0390
    106/157, train_loss: 0.0598
    107/157, train_loss: 0.0742
    108/157, train_loss: 0.0395
    109/157, train_loss: 0.0509
    110/157, train_loss: 0.0751
    111/157, train_loss: 0.0609
    112/157, train_loss: 0.0521
    113/157, train_loss: 0.0465
    114/157, train_loss: 0.0432
    115/157, train_loss: 0.0612
    116/157, train_loss: 0.0568
    117/157, train_loss: 0.0710
    118/157, train_loss: 0.0559
    119/157, train_loss: 0.0505
    120/157, train_loss: 0.0510
    121/157, train_loss: 0.0498
    122/157, train_loss: 0.0557
    123/157, train_loss: 0.0386
    124/157, train_loss: 0.0586
    125/157, train_loss: 0.0423
    126/157, train_loss: 0.0433
    127/157, train_loss: 0.0770
    128/157, train_loss: 0.0465
    129/157, train_loss: 0.0621
    130/157, train_loss: 0.0510
    131/157, train_loss: 0.0534
    132/157, train_loss: 0.0546
    133/157, train_loss: 0.0647
    134/157, train_loss: 0.0577
    135/157, train_loss: 0.0550
    136/157, train_loss: 0.0396
    137/157, train_loss: 0.0409
    138/157, train_loss: 0.0565
    139/157, train_loss: 0.0600
    140/157, train_loss: 0.0376
    141/157, train_loss: 0.0658
    142/157, train_loss: 0.0392
    143/157, train_loss: 0.0607
    144/157, train_loss: 0.0524
    145/157, train_loss: 0.0482
    146/157, train_loss: 0.0600
    147/157, train_loss: 0.0695
    148/157, train_loss: 0.0483
    149/157, train_loss: 0.0417
    150/157, train_loss: 0.0529
    151/157, train_loss: 0.0595
    152/157, train_loss: 0.0427
    153/157, train_loss: 0.0392
    154/157, train_loss: 0.0445
    155/157, train_loss: 0.0412
    156/157, train_loss: 0.0796
    157/157, train_loss: 0.0418
    158/157, train_loss: 0.0538
    epoch 3 average loss: 0.0647
    saved new best metric model
    current epoch: 3 current AUC: 1.0000 current accuracy: 0.9917 best AUC: 1.0000 at epoch: 3
    ----------
    epoch 4/4
    1/157, train_loss: 0.0300
    2/157, train_loss: 0.0368
    3/157, train_loss: 0.0387
    4/157, train_loss: 0.0494
    5/157, train_loss: 0.0348
    6/157, train_loss: 0.0519
    7/157, train_loss: 0.0453
    8/157, train_loss: 0.0338
    9/157, train_loss: 0.0556
    10/157, train_loss: 0.0546
    11/157, train_loss: 0.0499
    12/157, train_loss: 0.0272
    13/157, train_loss: 0.0398
    14/157, train_loss: 0.0441
    15/157, train_loss: 0.0498
    16/157, train_loss: 0.0422
    17/157, train_loss: 0.0338
    18/157, train_loss: 0.0750
    19/157, train_loss: 0.0508
    20/157, train_loss: 0.0583
    21/157, train_loss: 0.0425
    22/157, train_loss: 0.0456
    23/157, train_loss: 0.0421
    24/157, train_loss: 0.0571
    25/157, train_loss: 0.0477
    26/157, train_loss: 0.0625
    27/157, train_loss: 0.0542
    28/157, train_loss: 0.0519
    29/157, train_loss: 0.0424
    30/157, train_loss: 0.0378
    31/157, train_loss: 0.0382
    32/157, train_loss: 0.0441
    33/157, train_loss: 0.0394
    34/157, train_loss: 0.0724
    35/157, train_loss: 0.0305
    36/157, train_loss: 0.0452
    37/157, train_loss: 0.0510
    38/157, train_loss: 0.0426
    39/157, train_loss: 0.0376
    40/157, train_loss: 0.0536
    41/157, train_loss: 0.0399
    42/157, train_loss: 0.0354
    43/157, train_loss: 0.0479
    44/157, train_loss: 0.0349
    45/157, train_loss: 0.0501
    46/157, train_loss: 0.0355
    47/157, train_loss: 0.0528
    48/157, train_loss: 0.0457
    49/157, train_loss: 0.0450
    50/157, train_loss: 0.0391
    51/157, train_loss: 0.0420
    52/157, train_loss: 0.0327
    53/157, train_loss: 0.0507
    54/157, train_loss: 0.0391
    55/157, train_loss: 0.0457
    56/157, train_loss: 0.0274
    57/157, train_loss: 0.0424
    58/157, train_loss: 0.0302
    59/157, train_loss: 0.0434
    60/157, train_loss: 0.0531
    61/157, train_loss: 0.0376
    62/157, train_loss: 0.0384
    63/157, train_loss: 0.0380
    64/157, train_loss: 0.0411
    65/157, train_loss: 0.0280
    66/157, train_loss: 0.0379
    67/157, train_loss: 0.0341
    68/157, train_loss: 0.0385
    69/157, train_loss: 0.0313
    70/157, train_loss: 0.0540
    71/157, train_loss: 0.0306
    72/157, train_loss: 0.0378
    73/157, train_loss: 0.0327
    74/157, train_loss: 0.0270
    75/157, train_loss: 0.0438
    76/157, train_loss: 0.0409
    77/157, train_loss: 0.0371
    78/157, train_loss: 0.0296
    79/157, train_loss: 0.0379
    80/157, train_loss: 0.0262
    81/157, train_loss: 0.0418
    82/157, train_loss: 0.0528
    83/157, train_loss: 0.0225
    84/157, train_loss: 0.0564
    85/157, train_loss: 0.0393
    86/157, train_loss: 0.0298
    87/157, train_loss: 0.0271
    88/157, train_loss: 0.0435
    89/157, train_loss: 0.0547
    90/157, train_loss: 0.0360
    91/157, train_loss: 0.0392
    92/157, train_loss: 0.0293
    93/157, train_loss: 0.0606
    94/157, train_loss: 0.0350
    95/157, train_loss: 0.0280
    96/157, train_loss: 0.0347
    97/157, train_loss: 0.0415
    98/157, train_loss: 0.0356
    99/157, train_loss: 0.0505
    100/157, train_loss: 0.0360
    101/157, train_loss: 0.0430
    102/157, train_loss: 0.0335
    103/157, train_loss: 0.0231
    104/157, train_loss: 0.0413
    105/157, train_loss: 0.0260
    106/157, train_loss: 0.0375
    107/157, train_loss: 0.0288
    108/157, train_loss: 0.0321
    109/157, train_loss: 0.0343
    110/157, train_loss: 0.0406
    111/157, train_loss: 0.0323
    112/157, train_loss: 0.0406
    113/157, train_loss: 0.0335
    114/157, train_loss: 0.0262
    115/157, train_loss: 0.0288
    116/157, train_loss: 0.0216
    117/157, train_loss: 0.0377
    118/157, train_loss: 0.0299
    119/157, train_loss: 0.0556
    120/157, train_loss: 0.0283
    121/157, train_loss: 0.0310
    122/157, train_loss: 0.0326
    123/157, train_loss: 0.0386
    124/157, train_loss: 0.0220
    125/157, train_loss: 0.0225
    126/157, train_loss: 0.0431
    127/157, train_loss: 0.0316
    128/157, train_loss: 0.0368
    129/157, train_loss: 0.0427
    130/157, train_loss: 0.0390
    131/157, train_loss: 0.0453
    132/157, train_loss: 0.0276
    133/157, train_loss: 0.0395
    134/157, train_loss: 0.0232
    135/157, train_loss: 0.0235
    136/157, train_loss: 0.0320
    137/157, train_loss: 0.0326
    138/157, train_loss: 0.0252
    139/157, train_loss: 0.0309
    140/157, train_loss: 0.0199
    141/157, train_loss: 0.0285
    142/157, train_loss: 0.0236
    143/157, train_loss: 0.0365
    144/157, train_loss: 0.0353
    145/157, train_loss: 0.0381
    146/157, train_loss: 0.0333
    147/157, train_loss: 0.0377
    148/157, train_loss: 0.0247
    149/157, train_loss: 0.0357
    150/157, train_loss: 0.0327
    151/157, train_loss: 0.0238
    152/157, train_loss: 0.0183
    153/157, train_loss: 0.0431
    154/157, train_loss: 0.0352
    155/157, train_loss: 0.0445
    156/157, train_loss: 0.0222
    157/157, train_loss: 0.0246
    158/157, train_loss: 0.0596
    epoch 4 average loss: 0.0386
    saved new best metric model
    current epoch: 4 current AUC: 1.0000 current accuracy: 0.9963 best AUC: 1.0000 at epoch: 4
    train completed, best_metric: 1.0000 at epoch: 4


## Plot the loss and metric


```python
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val AUC")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
```


    
![Jupyter Notebook Plot](/assets/notebooks/2022-09-23-Monai_mednist_tutorial_files/2022-09-23-Monai_mednist_tutorial_28_0.png)
    


## Evaluate the model on test dataset

After training and validation, we already got the best model on validation test.  
We need to evaluate the model on test dataset to check whether it's robust and not over-fitting.  
We'll use these predictions to generate a classification report.


```python
model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())
```


```python
print(classification_report(
    y_true, y_pred, target_names=class_names, digits=4))
```

                  precision    recall  f1-score   support
    
       AbdomenCT     0.9919    0.9879    0.9899       995
       BreastMRI     0.9977    0.9920    0.9949       880
             CXR     1.0000    0.9949    0.9974       982
         ChestCT     0.9931    1.0000    0.9966      1014
            Hand     0.9952    0.9962    0.9957      1048
          HeadCT     0.9929    0.9990    0.9959       976
    
        accuracy                         0.9951      5895
       macro avg     0.9951    0.9950    0.9951      5895
    weighted avg     0.9951    0.9951    0.9951      5895
    


## Cleanup data directory

Remove directory if a temporary was used.


```python
if directory is None:
    shutil.rmtree(root_dir)
```

### GPU Utilization

|    0   N/A  N/A   2429330      C   /opt/conda/bin/python3.8         3799MiB |
+-----------------------------------------------------------------------------+
Fri Sep 23 17:26:34 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   39C    P0   244W / 400W |   3802MiB / 81251MiB |     50%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

