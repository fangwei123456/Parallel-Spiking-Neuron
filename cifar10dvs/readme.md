## Dependencies

The codes in this repo are modified from the TEBN paper [Temporal Effective Batch Normalization in Spiking Neural Networks](https://openreview.net/forum?id=fLIgyyQiJqz).

The codes in this repo require a specific modified SpikingJelly, which is provided in `./sj`.  The modification is implemented by the TEBN paper. More specifically, the authors of the TEBN paper add a flag `train` in the CIFAR10DVS dataset to control whether use the train set or the test set. While in the original SpikingJelly, the train set and the test set are split from the original dataset by a function.

Install this specific SpikingJelly:

```bash
cd ./sj

python setup.py install
```

The CIFAR10DVS dataset in this specific SpikingJelly requires a `train` and a `test` directory in the dataset directory. Here is an example:

```bash
(base) root@ubuntu:/datasets/CIFAR10DVS/frames_number_10_split_by_number$ ls
test  train
(base) root@ubuntu:/datasets/CIFAR10DVS/frames_number_10_split_by_number/train$ ls
airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
(base) root@ubuntu:/datasets/CIFAR10DVS/frames_number_10_split_by_number/test$ ls
airplane  automobile  bird  cat  deer  dog  frog  horse  ship  truck
```

We use the same split method as the TEBN paper, which is using `0-99` in each class as the test set, and the `100-999` as the train set. Such a split can be done by `./move_data.py`.

## Usage

```
usage: train_vgg.py [-h] [-j N] [--epochs N] [--start_epoch N] [-b N] [--lr LR] [--seed SEED] [-T N] [--means N] [--lamb N]
                    [-out_dir OUT_DIR] [-resume RESUME] [-method METHOD] [-opt OPT] [-tau TAU] [-TET]

PyTorch Training

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 10)
  --epochs N            number of total epochs to run
  --start_epoch N       manual epoch number (useful on restarts)
  -b N, --batch_size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel
  --lr LR, --learning_rate LR
                        initial learning rate
  --seed SEED           seed for initializing training.
  -T N                  snn simulation time (default: 2)
  --means N             make all the potential increment around the means (default: 1.0)
  --lamb N              adjust the norm factor to avoid outlier (default: 0.0)
  -out_dir OUT_DIR      root dir for saving logs and checkpoint
  -resume RESUME        resume from the checkpoint path
  -method METHOD        use which network
  -opt OPT              optimizer method
  -tau TAU              tau of LIF
  -TET                  use the tet loss
```

The options used in the paper are

```
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 4
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 8
python train_vgg.py -b 32 --epochs 200 -method PSN -TET -T 10
```

And the number of GPUs is 2.

The original terminal outputs are saved in

```
T4_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu.log
T8_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu.log
T10_opt_SGD0.1_tau_0.25_method_PSN_b_32_TET_2gpu.log
```

