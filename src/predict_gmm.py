# coding=utf-8
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image, ImageDraw
import numpy as np

import argparse
import time
import json
from networks import GMM, load_checkpoint

from visualization import save_images


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--grid_size", type=int, default=5)

    parser.add_argument(
        "--result_dir", type=str, default="output", help="save result infos"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/GMM/gmm_final.pth",
        help="model checkpoint for test",
    )
    parser.add_argument("--person-image", help="path to person image", required=True)
    parser.add_argument("--person-mask-image", help="path to person segmentation mask", required=True)
    parser.add_argument("--person-parse-image", help="path to person parsing output image", required=True)
    parser.add_argument("--cloth-image", help="path to cloth image", required=True)
    parser.add_argument("--cloth-mask-image", help="path to cloth mask image", required=True)
    parser.add_argument("--pose-file", help="path to openpose annotation for person", required=True)

    opt = parser.parse_args()
    return opt


def predict_gmm(opt, model, inputs, save_intermediate=False):
    model.cuda()
    model.eval()
    person_image = inputs["image"].cuda()
    cloth_mask = inputs["cloth_mask"].unsqueeze(0).cuda()
    cloth_orig = inputs["cloth"].unsqueeze(0).cuda()
    image_grid = inputs["grid_image"].unsqueeze(0).cuda()
    agnostic = inputs["agnostic"].unsqueeze(0).cuda()

    grid, theta = model(agnostic, cloth_mask)
    warped_cloth = F.grid_sample(cloth_orig, grid, padding_mode="border")
    warped_mask = F.grid_sample(cloth_mask, grid, padding_mode="zeros")
    warped_grid = F.grid_sample(image_grid, grid, padding_mode="zeros")
    overlay = 0.7 * warped_cloth + 0.3 * person_image
    if save_intermediate:
        save_images(warped_cloth, ["warped_cloth.jpg"], "output/debug")
        save_images(warped_mask, ["warped_mask.jpg"], "output/debug")
        save_images(warped_grid, ["warped_grid.jpg"], "output/debug")
        save_images(overlay, ["overlay.jpg"], "output/debug")

    return overlay


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
        + (parse_array == 4).astype(np.float32)
        + (parse_array == 13).astype(np.float32)
    )
    parse_cloth = (5 == parse_array).astype(np.float32) + \
        (6 == parse_array).astype(np.float32) + \
        (7 == parse_array).astype(np.float32)

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
    grid_image = Image.open("grid.png")
    grid_tensor = transform(grid_image)
    parse_cloth_tensor.unsqueeze_(0)

    result = {
        "c_name": cloth_image_path,
        "im_name": person_image_path,
        "cloth": cloth_tensor,
        "cloth_mask": cloth_mask_tensor,
        "image": person_tensor,
        "agnostic": agnostic,
        "grid_image": grid_tensor,
    }
    return result


def main():
    opt = get_opt()
    print(opt)

    # create model & test
    model = GMM(opt)
    load_checkpoint(model, opt.checkpoint)
    with torch.no_grad():
        inputs = prepare_inputs(opt)
        st = time.time()
        predict_gmm(opt, model, inputs, save_intermediate=True)
        print("time: ", time.time() - st)


if __name__ == "__main__":
    main()
