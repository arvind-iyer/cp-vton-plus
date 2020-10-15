import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import json
import numpy as np

from networks import GMM, UnetGenerator, load_checkpoint


def make_image(img_tensors):
    for img_tensor in img_tensors:
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)
        array = tensor.numpy().astype("uint8")
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        return Image.fromarray(array)


class GMMEngine:
    def __init__(
        self, checkpoint_path=os.path.join(os.getenv("MODEL_PATH", ""), "gmm_final.pth")
    ):
        self.name = "GMM"
        self.b = 1  # batch size
        self.stage = "GMM"
        self.fine_width = 192
        self.fine_height = 256
        self.radius = 5
        self.display_count = 1
        self.datamode = "test"
        self.grid_size = 5
        self.checkpoint = checkpoint_path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load(self):
        self.model = GMM(self)
        load_checkpoint(self.model, self.checkpoint)

    def prepare_inputs(
        self,
        person_image,
        person_parse_image,
        person_mask_image,
        cloth_image,
        cloth_mask_image,
        pose_data_dict: dict,
    ):
        cloth_tensor = self.transform(cloth_image)
        input_cloth_mask = (np.array(cloth_mask_image) >= 128).astype(np.float32)
        cloth_mask_tensor = torch.from_numpy(input_cloth_mask).unsqueeze(0)
        person_tensor = self.transform(person_image)

        parse_array = np.array(person_parse_image)

        parse_head = (
            (parse_array == 1).astype(np.float32)
            + (parse_array == 4).astype(np.float32)
            + (parse_array == 13).astype(np.float32)
        )
        parse_cloth = (
            (5 == parse_array).astype(np.float32)
            + (6 == parse_array).astype(np.float32)
            + (7 == parse_array).astype(np.float32)
        )

        parse_mask_array = (np.array(person_mask_image) > 0).astype(np.float32)

        # downsample person parsing
        parse_mask_orig = Image.fromarray((parse_mask_array * 256).astype(np.uint8))
        parse_mask_small = parse_mask_orig.resize(
            (self.fine_width // 16, self.fine_height // 16), Image.BILINEAR
        )
        parse_mask_array = parse_mask_small.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR
        )

        parse_mask_orig = parse_mask_orig.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR
        )

        parse_mask_tensor = self.transform(parse_mask_array)

        parse_head_tensor = torch.from_numpy(parse_head)
        parse_cloth_tensor = torch.from_numpy(parse_cloth)
        parse_cloth_tensor.unsqueeze_(0)

        # upper cloth
        image_head = person_tensor * parse_head_tensor - (1 - parse_head_tensor)

        pose_array = np.array(pose_data_dict["people"][0]["pose_keypoints"]).reshape(
            (-1, 3)
        )
        point_num = pose_array.shape[0]
        pose_map = torch.zeros(point_num, self.fine_width, self.fine_height)

        pose_image = Image.new("L", (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(pose_image)

        for i in range(point_num):
            one_map = Image.new("L", (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            px, py = pose_array[i, :2]
            if px > 1 and py > 1:
                draw.rectangle(
                    (
                        px - self.radius,
                        py - self.radius,
                        px + self.radius,
                        py + self.radius,
                    ),
                    "white",
                    "white",
                )
                pose_draw.rectangle(
                    (
                        px - self.radius,
                        py - self.radius,
                        px + self.radius,
                        py + self.radius,
                    ),
                    "white",
                    "white",
                )
            pose_map[i] = self.transform(one_map)[0]

        agnostic = torch.cat([parse_mask_tensor, image_head, pose_map], 0)
        grid_image = Image.open("grid.png")
        grid_tensor = self.transform(grid_image)

        return {
            "cloth": cloth_tensor,
            "shape": parse_mask_tensor,
            "cloth_mask": cloth_mask_tensor,
            "image": person_tensor,
            "agnostic": agnostic,
            "grid_image": grid_tensor,
        }

    def predict(self, inputs):
        self.model.cuda()
        self.model.eval()

        cloth_mask = inputs["cloth_mask"].unsqueeze(0).cuda()
        cloth_orig = inputs["cloth"].unsqueeze(0).cuda()
        agnostic = inputs["agnostic"].unsqueeze(0).cuda()

        grid, theta = self.model(agnostic, cloth_mask)
        warped_cloth = F.grid_sample(cloth_orig, grid, padding_mode="border")
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode="zeros")

        inputs["cloth"] = warped_cloth
        inputs["cloth_mask"] = warped_mask
        return inputs


class TOMEngine:
    def __init__(
        self, checkpoint_path=os.path.join(os.getenv("MODEL_PATH", ""), "tom_final.pth")
    ):
        self.name = "TOM"
        self.workers = 1
        self.batch_size = 1
        self.stage = "TOM"
        self.fine_width = 192
        self.fine_height = 256
        self.datamode = "test"
        self.radius = 5
        self.checkpoint = checkpoint_path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load(self):
        self.model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(self.model, self.checkpoint)

    def prepare_inputs(
        self,
        person_image,
        person_parse_image,
        person_mask_image,
        cloth_image,
        cloth_mask_image,
        pose_data,
    ):
        cloth_tensor = self.transform(cloth_image)
        input_cloth_mask = (np.array(cloth_mask_image) >= 128).astype(np.float32)
        cloth_mask_tensor = torch.from_numpy(input_cloth_mask).unsqueeze(0)
        person_tensor = self.transform(person_image)

        person_tensor = self.transform(person_image)

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

        person_mask_array = np.array(person_mask_image)
        parse_shape = (person_mask_array > 0).astype(np.float32)

        # downsample shape
        parse_shape_orig = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize(
            (self.fine_width // 16, self.fine_height // 16), Image.BILINEAR
        )
        parse_shape_orig = parse_shape_orig.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR
        )
        shape_tensor = self.transform(parse_shape)

        parse_head_tensor = torch.from_numpy(parse_head)
        parse_cloth_tensor = torch.from_numpy(parse_cloth)

        # upper cloth
        image_head = person_tensor * parse_head_tensor - (1 - parse_head_tensor)

        pose_array = np.array(pose_data["people"][0]["pose_keypoints"]).reshape((-1, 3))

        point_num = pose_array.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        pose_image = Image.new("L", (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(pose_image)

        for i in range(point_num):
            one_map = Image.new("L", (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)

            px = pose_data[i, 0]
            py = pose_data[i, 1]

            if px > 1 and py > 1:
                draw.rectangle(
                    (
                        px - self.radius,
                        py - self.radius,
                        px + self.radius,
                        py + self.radius,
                    ),
                    "white",
                    "white",
                )
                pose_draw.rectangle(
                    (
                        px - self.radius,
                        py - self.radius,
                        px + self.radius,
                        py + self.radius,
                    ),
                    "white",
                    "white",
                )

            pose_map[i] = self.transform(one_map)[0]
        agnostic = torch.cat([shape_tensor, image_head, pose_map], 0)

        parse_cloth_tensor.unsqueeze_(0)

        return {
            "shape": shape_tensor,
            "head": image_head,
            "cloth": cloth_tensor,
            "cloth_mask": cloth_mask_tensor,
            "agnostic": agnostic,
        }

    def predict(self, inputs):
        self.model.cuda()
        self.model.eval()

        agnostic = inputs["agnostic"].unsqueeze(0).cuda()
        cloth_tensor = inputs["cloth"].unsqueeze(0).cuda()
        cloth_mask_tensor = inputs["cloth_mask"].unsqueeze(0).cuda()

        outputs = self.model(torch.cat([agnostic, cloth_tensor, cloth_mask_tensor], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = cloth_tensor * m_composite + p_rendered + (1 - m_composite)

        tryon_image = make_image(p_tryon)
        return tryon_image


def load_cloth_image(outfit_id):
    # TODO: load images from a db service instead
    CLOTH_DIR = "../mydata/test/cloth"
    CLOTH_MASK_DIR = "../mydata/test/cloth-mask"
    cloth_image = Image.open(os.path.join(CLOTH_DIR, f"{outfit_id}.jpg"))
    cloth_mask_image = Image.open(os.path.join(CLOTH_MASK_DIR, f"{outfit_id}.jpg"))
    return cloth_image, cloth_mask_image


def load_parse_image():
    """TODO: Replace with code that runs JPPNet
    """
    return Image.open("../mydata/test/image-parse/000001_0_parse.png")


def load_person_mask_image():
    """TODO: Replace with code that runs JPPNet
    """
    return Image.open("../mydata/test/image-parse/000001_0_parse.png")


class InferenceEngine:
    def __init__(self, gmm_model_path, tom_model_path):
        self.gmm = GMMEngine(gmm_model_path)
        self.tom = TOMEngine(tom_model_path)
        pass

    def load(self):
        pass

    def infer(self, image_file, outfit_id, pose_data):
        # load image
        try:
            person_image = Image.open(image_file).convert("RGB")
        except Exception:
            raise Exception("Invalid image input data")

        try:
            pose_data_dict = json.load(pose_data)
            # TODO: validate pose array
        except Exception:
            raise Exception("Invalid pose input data")

        # get cloth image and mask using the outfit_id
        cloth_image, cloth_mask_image = load_cloth_image(outfit_id)
        # run person segmentation
        # TODO: create a script to run LIP_JPPNet
        person_parse_image = load_parse_image()
        # get binary body mask
        person_mask_image = load_person_mask_image()
        # process neck segment
        # run gmm
        inputs = self.gmm.prepare_inputs(
            person_image,
            person_parse_image,
            person_mask_image,
            cloth_image,
            cloth_mask_image,
            pose_data_dict,
        )
        # object containing preprocessed person and pose information
        # with the warped clothes tensor
        gmm_out = self.gmm.predict(inputs)
        # get the warped cloth image
        # run tom
        # return try-on image
        output = self.tom.predict(gmm_out)
        return output
