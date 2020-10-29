from collections import OrderedDict

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

from .resnet import resnet101
from .transforms import get_affine_transform, transform_parsing


def _xywh2cs(x, y, w, h):
    aspect_ratio = 1.0
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + (w * 0.5)
    center[1] = y + (h * 0.5)
    if w > aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
    return center, scale


def preprocess(image, transform, crop_size=(473, 473)):
    h, w = image.shape[:2]
    person_center, s = _xywh2cs(0, 0, w - 1, h - 1)
    r = 0
    affine_transform = get_affine_transform(person_center, s, r, crop_size)
    inputs = cv2.warpAffine(image,
                            affine_transform,
                            crop_size,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
    print(transform)
    inputs_batch = transform(inputs) 

    return inputs_batch, {
        "center": person_center,
        "height": h,
        "width": w,
        "scale": s,
        "rotation": r
    }


def load_model(model_path):
    model = resnet101(num_classes=20, pretrained=None)
    # Load model weights
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()
    return model


def multi_scale_testing(model,
                        batch_input_im,
                        crop_size=[473, 473],
                        flip=True,
                        multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size,
                               mode='bilinear',
                               align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s,
                                      mode='bilinear',
                                      align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1]
        output = parsing_output[0]
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output.unsqueeze(0))
        ms_outputs.append(output[0])
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)
    ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=2)
    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()
    return parsing, ms_fused_parsing_output




def predict(model, image, transform=None):
    if transform is None:
        transform = get_transform(model)
    image_batch, metadata = preprocess(np.array(image), transform)
    if len(image_batch.shape) > 4:
        image_batch = image_batch.squeeze()
    
    c = metadata['center']
    s = metadata['scale']
    w = metadata['width']
    h = metadata['height']
    input_size = (473, 473)

    parsing, logits = multi_scale_testing(model, 
                                          image_batch.cuda(),
                                          crop_size=input_size,
                                          flip=False,
                                          multi_scales=[1.0])

    parsing_result = transform_parsing(parsing, c, s, w, h, input_size)
    output_im = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
    return output_im


def get_transform(model, input_space='BGR'):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=model.mean, std=model.std),
    ])
