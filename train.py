
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import torch.optim as optim
import torch.nn as nn
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import  OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
from dlvc.trainer_reference_impl import ImgSemSegTrainer

"""
1. 
Method train performs image transformations on train and val sets.
Train and vala data are loaded from the specified paths. Val set is used as test set.
Device type is specified.
Model, optimizer, loss, and lr scheduler are left to be specified, too.
Train calls trainer.py.
Argparse is used for running the script.
"""

def train(args):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    train_data = OxfordPetsCustom(root="dlvc/datasets", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root="dlvc/datasets", 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization of a model 
    # 1) from scratch
    model_net = fcn_resnet50(weights=None)

    # 2) with pretrained encoder
    # model_net = fcn_resnet50(weights_backbone=FCN_ResNet50_Weights.DEFAULT)

    num_classes = len(train_data.classes_seg) # Adjust number of classes
    in_features = model_net.classifier[4].in_channels
    model_net.classifier[4] = nn.Conv2d(in_features, num_classes, kernel_size=1)

    model = DeepSegmenter(model_net).to(device) # DeepSegmenter is a wrapper
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
    loss_fn = nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trainer_reference_impl = ImgSemSegTrainer(model, # instead of train
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    #trainer.train()
    trainer_reference_impl.train()

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    # trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 3 # add 30 epochs later


    train(args)