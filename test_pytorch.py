from __future__ import division
from __future__ import print_function

from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import settings

# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    """ Resnet18
    """
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224


    return model_ft, input_size



# Number of classes in the dataset
num_classes = 2

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

device = torch.device("cuda:0" if settings.USE_GPU_MASK_DETECTOR else "cpu")

model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
model_ft.eval()
model_ft.to(device)

model_ft.load_state_dict(torch.load(
    settings.custom_face_mask_detector)[
                             'model_state_dict'])

data_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def detect_face_mask(image):
    img = cv2.cvtColor(deepcopy(image), cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img)

    img = data_transform(image_pil)
    img = img.unsqueeze(0).to(device)
    out = model_ft(img).cpu().detach().tolist()[0]
    mask, withoutmask = torch.nn.functional.softmax(torch.tensor(out)).tolist()
    # print(mask, withoutmask)
    return True if mask > withoutmask else False, mask


if __name__ == "__main__":
    from src import myutils
    main_path = Path("/home/meditab/Desktop/test/crops")
    filtered= []
    result = []
    for img_path in main_path.rglob('*.png'):
        img = cv2.imread(str(img_path))
        label, mask_prob = detect_face_mask(img)
        cv2.putText(img, str(mask_prob), (5, 15), 0, 0.75, (50, 200, 0), 2)
        # cv2.imshow("result", img)
        # cv2.waitKey(0)
        # print(img_path)
        if label:
            filtered.append(img)
        else:
            result.append(img)
    path = main_path.parent / "mask"
    path.mkdir(parents=True, exist_ok=True)
    myutils.save_images(str(path), filtered)

    path = main_path.parent / "no mask"
    path.mkdir(parents=True, exist_ok=True)
    myutils.save_images(str(path), result)