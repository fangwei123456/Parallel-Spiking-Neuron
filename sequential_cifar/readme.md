## Usage

`glif.py` in this repo is modified from the GLIF paper [GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks](https://openreview.net/forum?id=UmFSx2c4ubT).

Note that the name of neurons in the paper are not identical to those in the codes. Here is the name table:

| Neuron in the paper | Neuron in the codes |
| ------------------- | ------------------- |
| PSN                 | if5                 |
| masked PSN          | if5pmd8             |
| sliding PSN         | mspsn               |
| LIF wo reset        | lifnr               |



```
usage: train_secf10.py [-h] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                       [-resume RESUME] [-amp] [-opt OPT] [-momentum MOMENTUM] [-lr LR] [-channels CHANNELS]
                       [-neu NEU] [-class-num CLASS_NUM] [-P P] [-exp-init]

Classify Sequential CIFAR10/100

options:
  -h, --help            show this help message and exit
  -device DEVICE        device
  -b B                  batch size
  -epochs N             number of total epochs to run
  -j N                  number of data loading workers (default: 4)
  -data-dir DATA_DIR    root dir of CIFAR10/100 dataset
  -out-dir OUT_DIR      root dir for saving logs and checkpoint
  -resume RESUME        resume from the checkpoint path
  -amp                  automatic mixed precision training
  -opt OPT              use which optimizer. SDG or Adam
  -momentum MOMENTUM    momentum for SGD
  -lr LR                learning rate
  -channels CHANNELS    channels of CSNN
  -neu NEU              use which neuron
  -class-num CLASS_NUM
  -P P                  the order of the masked/sliding PSN
  -exp-init             use the exp init method to initialize the weight of SPSN
```

### Examples

Use the PSN on sequential CIFAR10:

```bash
python train_secf10.py -data-dir /userhome/datasets/CIFAR10 -amp -opt sgd -channels 128 -epochs 256 -neu if5
```

Use the 32-order SPSN on sequential CIFAR100:

```bash
python train_secf10.py -data-dir /userhome/datasets/CIFAR100 -class-num 100 -amp -opt adamw -lr 0.001 -channels 128 -epochs 256 -neu mspsn -P 32
```

Use the 32-order masked PSN on sequential CIFAR100:

```
python train_secf10.py -data-dir /userhome/datasets/CIFAR100 -class-num 100 -amp -opt sgd -channels 128 -epochs 256 -neu if5pmd8 -P 32
```

