## Usage

```bash
usage: train_cf10.py [-h] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                     [-resume RESUME] [-amp] [-opt OPT] [-momentum MOMENTUM] [-lr LR] [-channels CHANNELS] [-T T]

Classify CIFAR10

options:
  -h, --help          show this help message and exit
  -device DEVICE      device
  -b B                batch size
  -epochs N           number of total epochs to run
  -j N                number of data loading workers (default: 4)
  -data-dir DATA_DIR  root dir of the CIFAR10 dataset
  -out-dir OUT_DIR    root dir for saving logs and checkpoint
  -resume RESUME      resume from the checkpoint path
  -amp                automatic mixed precision training
  -opt OPT            use which optimizer. SDG or AdamW
  -momentum MOMENTUM  momentum for SGD
  -lr LR              learning rate
  -channels CHANNELS  channels of CSNN
  -T T                number of time-steps
```

The options used in the paper are:

```
python train_cf10.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -device cuda:0 -T 4
```

The original training args are:

```
Namespace(device='cuda:1', b=128, epochs=1024, j=4, data_dir='/datasets/CIFAR10', out_dir='./logs_cf10', resume=None, amp=True, opt='sgd', momentum=0.9, lr=0.1, channels=256, T=4, neu='if5', sg='atan')
train_cf10.py -data-dir /datasets/CIFAR10 -amp -opt sgd -channels 256 -epochs 1024 -neu if5 -device cuda:1 -T 4
```

Note that the PSN is named IF5 in the codes.
