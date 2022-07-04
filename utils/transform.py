import torchvision.transforms as transforms
import numpy as np
import torch



"""

attention_train_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     transforms.ColorJitter(brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.3
                            ),
     # transforms.CenterCrop((64,64)),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.RandomPerspective(distortion_scale=0.2,p=0.3),
     transforms.RandomResizedCrop(size=(64,64), scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),

     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Lambda(lambda crops: torch.unsqueeze(crops, 0)),
     # transforms.RandomErasing()
     ])


attention_test_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     # transforms.Resize((73,73)),
     # transforms.CenterCrop((64, 64)),

        # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

"""

attention_train_transform = transforms.Compose(
    [
     transforms.Resize((73,73)),
     transforms.RandomCrop((64,64)),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])


attention_test_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
        # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])





cnn_train_transform = transforms.Compose(
    [
     transforms.RandomCrop((64,64)),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])

cnn_test_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

mid_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

cnn_points_train_transform = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda crops:
               transforms.FiveCrop((38,38))(crops)+transforms.FiveCrop((55,55))(crops)),

        transforms.Lambda(lambda crops:
               [mid_transform(crop) for crop in crops]),
    ])

cnn_points_test_transform = transforms.Compose(
    [
        transforms.Resize((64,64)),

        transforms.Lambda(lambda crops:
               transforms.FiveCrop((38,38))(crops)+transforms.FiveCrop((55,55))(crops)),

        transforms.Lambda(lambda crops:
               [mid_transform(crop) for crop in crops]),
    ])
