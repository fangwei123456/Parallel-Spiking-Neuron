import os
import shutil
class_num = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

root = '/datasets/CIFAR10DVS/frames_number_10_split_by_number'

for cn in class_num:
    source = os.path.join(root, cn)
    target = os.path.join(root, 'test', cn)
    if not os.path.exists(target):
        os.makedirs(target)

    for i in range(100):
        os.symlink(os.path.join(source, f'cifar10_{cn}_{i}.npz'), os.path.join(target, f'cifar10_{cn}_{i}.npz'))

    target = os.path.join(root, 'train', cn)
    if not os.path.exists(target):
        os.makedirs(target)
    for i in range(100, 1000):

        os.symlink(os.path.join(source, f'cifar10_{cn}_{i}.npz'), os.path.join(target, f'cifar10_{cn}_{i}.npz'))

