## Usage

The codes in this repo are modified from the SEW ResNet paper [Deep Residual Learning in Spiking Neural Networks](https://arxiv.org/abs/2102.04159).

```bash
usage: train.py [-h] [--data-path DATA_PATH] [--model MODEL] [--device DEVICE] [-b BATCH_SIZE] [--epochs N] [-j N]
                [--lr LR] [--momentum M] [--wd W] [--print-freq PRINT_FREQ] [--output-dir OUTPUT_DIR]
                [--resume RESUME] [--start-epoch N] [--cache-dataset] [--sync-bn] [--test-only] [--amp]
                [--world-size WORLD_SIZE] [--dist-url DIST_URL] [--tb] [--T T] [--adamw] [--cos_lr_T COS_LR_T]
                [--load LOAD] [--tet]

PyTorch Classification Training

options:
  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        dataset
  --model MODEL         model
  --device DEVICE       device
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 16)
  --lr LR               initial learning rate
  --momentum M          Momentum for SGD. Adam will not use momentum
  --wd W, --weight-decay W
                        weight decay (default: 0)
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir OUTPUT_DIR
                        path where to save
  --resume RESUME       resume from checkpoint
  --start-epoch N       start epoch
  --cache-dataset       Cache the datasets for quicker initialization. It also serializes the transforms
  --sync-bn             Use sync batch norm
  --test-only           Only test the model
  --amp                 Use AMP training
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
  --tb                  Use TensorBoard to record logs
  --T T                 simulation steps
  --adamw               Use AdamW. The default optimizer is SGD.
  --cos_lr_T COS_LR_T   T_max of CosineAnnealingLR.
  --load LOAD           the pt file path for loading pre-trained ANN weights
  --tet                 use the tet loss
```

The options used in the paper are

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet34 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /dataset/ImageNet2012 --load /userhome/pretrained/resnet34-b627a593.pth --tet

python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /dataset/ImageNet2012 --load /userhome/pretrained/resnet18-f37072fd.pth --tet -b 64
```

The pre-trained ANN weights are available on pytorch.org:

    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth"
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth"
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth"
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth"

