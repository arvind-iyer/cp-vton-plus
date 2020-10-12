# coding=utf-8
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image, ImageDraw

import argparse
import os
import time
import json

from networks import UnetGenerator, load_checkpoint
from visualization import save_images


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="TOM", choices=["TOM"])

    parser.add_argument("--gpu_ids", default="")
    parser.add_argument("-j", "--workers", type=int, default=1)
    parser.add_argument("-b", "--batch-size", type=int, default=1)

    parser.add_argument("--stage", default="TOM", choices=["TOM"])

    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument(
        "--result_dir", type=str, default="output", help="save result infos"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/TOM/tom_final.pth",
        help="model checkpoint for test",
    )

    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--person-image", help="path to person image", required=True)
    parser.add_argument("--person-mask-image", help="path to person segmentation mask", required=True)
    parser.add_argument("--person-parse-image", help="path to person parsing output image", required=True)
    parser.add_argument("--cloth-image", help="path to cloth image", required=True)
    parser.add_argument("--cloth-mask-image", help="path to cloth mask image", required=True)
    parser.add_argument("--pose-file", help="path to openpose annotation for person", required=True)

    opt = parser.parse_args()
    return opt


def prepare_inputs(opt):
    person_image_path = opt.person_image
    person_parse_image_path = opt.person_parse_image
    person_mask_image_path = opt.person_mask_image
    cloth_image_path = opt.cloth_image
    cloth_mask_image_path = opt.cloth_mask_image
    pose_file_path = opt.pose_file
    FINE_WIDTH = opt.fine_width
    FINE_HEIGHT = opt.fine_height
    RADIUS = opt.radius
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cloth_image = Image.open(cloth_image_path)
    cloth_mask_image = Image.open(cloth_mask_image_path).convert("L")
    person_image = Image.open(person_image_path)
    person_parse_image = Image.open(person_parse_image_path).convert("L")

    cloth_tensor = transform(cloth_image)
    input_cloth_mask = np.array(cloth_mask_image)
    input_cloth_mask = (input_cloth_mask >= 128).astype(np.float32)
    cloth_mask_tensor = torch.from_numpy(input_cloth_mask)
    cloth_mask_tensor.unsqueeze_(0)
    person_tensor = transform(person_image)

    parse_array = np.array(person_parse_image)
    parse_head = (
        (parse_array == 1).astype(np.float32)
        + (parse_array == 2).astype(np.float32)
        + (parse_array == 4).astype(np.float32)
        + (parse_array == 9).astype(np.float32)
        + (parse_array == 12).astype(np.float32)
        + (parse_array == 13).astype(np.float32)
        + (parse_array == 16).astype(np.float32)
        + (parse_array == 17).astype(np.float32)
    )
    parse_cloth = (
        (5 == parse_array).astype(np.float32)
        + (6 == parse_array).astype(np.float32)
        + (7 == parse_array).astype(np.float32)
    )

    person_mask_image = Image.open(person_mask_image_path).convert("L")
    person_mask_array = np.array(person_mask_image)
    parse_shape = (person_mask_array > 0).astype(np.float32)

    # downsample shape
    parse_shape_orig = Image.fromarray((parse_shape * 255).astype(np.uint8))
    parse_shape = parse_shape_orig.resize(
        (FINE_WIDTH // 16, FINE_HEIGHT // 16), Image.BILINEAR
    )
    parse_shape = parse_shape.resize((FINE_WIDTH, FINE_HEIGHT), Image.BILINEAR)
    parse_shape_orig = parse_shape_orig.resize(
        (FINE_WIDTH, FINE_HEIGHT), Image.BILINEAR
    )
    shape_tensor = transform(parse_shape)
    parse_head_tensor = torch.from_numpy(parse_head)
    parse_cloth_tensor = torch.from_numpy(parse_cloth)

    # upper cloth
    image_head = person_tensor * parse_head_tensor - (1 - parse_head_tensor)

    # load pose
    with open(pose_file_path, "r") as f:
        pose_label = json.load(f)
        pose_data = pose_label["people"][0]["pose_keypoints"]
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, FINE_HEIGHT, FINE_WIDTH)
    pose_image = Image.new("L", (FINE_WIDTH, FINE_HEIGHT))
    pose_draw = ImageDraw.Draw(pose_image)

    for i in range(point_num):
        one_map = Image.new("L", (FINE_WIDTH, FINE_HEIGHT))
        draw = ImageDraw.Draw(one_map)
        px = pose_data[i, 0]
        py = pose_data[i, 1]
        if px > 1 and px > 1:
            draw.rectangle(
                (px - RADIUS, py - RADIUS, px + RADIUS, py + RADIUS), "white", "white"
            )
            pose_draw.rectangle(
                (px - RADIUS, py - RADIUS, px + RADIUS, py + RADIUS), "white", "white"
            )
        pose_map[i] = transform(one_map)[0]
    print("sizes: ", shape_tensor.shape, image_head.shape, pose_map.shape)
    agnostic = torch.cat([shape_tensor, image_head, pose_map], 0)
    print("agnostic", agnostic.shape)
    parse_cloth_tensor.unsqueeze_(0)

    result = {
        "c_name": cloth_image_path,
        "im_name": person_image_path,
        "shape": shape_tensor,
        "head": image_head,
        "cloth": cloth_tensor,
        "cloth_mask": cloth_mask_tensor,
        "pose_image": pose_image,
        "agnostic": agnostic,
    }
    return result


def test_tom(opt, model, inputs):
    model.cuda()
    model.eval()

    iter_start_time = time.time()

    im_pose = inputs["pose_image"]
    im_h = inputs["head"]
    shape = inputs["shape"]

    agnostic = inputs["agnostic"].unsqueeze(0).cuda()
    c = inputs["cloth"].unsqueeze(0).cuda()
    cm = inputs["cloth_mask"].unsqueeze(0).cuda()

    # outputs = model(torch.cat([agnostic, c], 1))  # CP-VTON
    outputs = model(torch.cat([agnostic, c, cm], 1))  # CP-VTON+
    p_rendered, m_composite = torch.split(outputs, 3, 1)
    p_rendered = F.tanh(p_rendered)
    m_composite = F.sigmoid(m_composite)
    p_tryon = c * m_composite + p_rendered * (1 - m_composite)

    # visuals = [[im_h, shape, im_pose],
    #            [c, 2*cm-1, m_composite],
    #            [p_rendered, p_tryon, im]]

    output_dir = os.path.join(opt.result_dir, "tom")
    save_images(p_tryon, ["tryon.jpg"], output_dir)
    save_images(im_h, ["image_head.jpg"], output_dir)
    save_images(shape, ["imshape.jpg"], output_dir)
    # save_images(im_pose, ["impose.jpg"], output_dir)
    save_images(m_composite, ["composite.jpg"], output_dir)
    save_images(p_rendered, ["rendered.jpg"], output_dir)  # For test data

    t = time.time() - iter_start_time
    print("time: %.3f" % (t,), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
    # create model & test
    model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON+
    load_checkpoint(model, opt.checkpoint)
    with torch.no_grad():
        test_tom(opt, model, prepare_inputs(opt))

    print("Finished test %s, named: %s!" % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
