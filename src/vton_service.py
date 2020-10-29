import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
import json
import numpy as np

from networks import GMM, UnetGenerator, load_checkpoint


DATA_DIR = os.getenv("DATA_DIR", "../data")


def make_image_array(img_tensors):
    for img_tensor in img_tensors:
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)
        array = tensor.detach().numpy().astype("uint8")
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        return array


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
        parse_mask_orig = Image.fromarray((parse_mask_array * 255).astype(np.uint8))
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
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)

        pose_image = Image.new("L", (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(pose_image)

        for i in range(point_num):
            one_map = Image.new("L", (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            px = pose_array[i, 0]
            py = pose_array[i, 1]
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
        grid_image = Image.open("../grid.png")
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

        person_image = inputs["image"].cuda()
        cloth_mask = inputs["cloth_mask"].unsqueeze(0).cuda()
        cloth_orig = inputs["cloth"].unsqueeze(0).cuda()
        agnostic = inputs["agnostic"].unsqueeze(0).cuda()
        image_grid = inputs["grid_image"].unsqueeze(0).cuda()

        grid, theta = self.model(agnostic, cloth_mask)
        warped_cloth = F.grid_sample(cloth_orig, grid, padding_mode="border")
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode="zeros")
        # warped_grid = F.grid_sample(image_grid, grid, padding_mode="zeros")
        overlay = 0.7 * warped_cloth + 0.3 * person_image
        Image.fromarray(make_image_array(overlay)).save("../output/test/overlay.jpg")
        Image.fromarray(make_image_array(warped_mask)).save("../output/test/warped_mask.jpg")

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
        print("load TOM weights from: ", self.checkpoint)
        load_checkpoint(self.model, self.checkpoint)

    def prepare_inputs(
        self,
        gmm_outputs,
        person_image,
        person_parse_image,
        person_mask_image,
        pose_data
    ):
        # cloth_tensor = self.transform(make_arra)
        # input_cloth_mask = (np.array(cloth_mask_image) >= 128).astype(np.float32)
        # cloth_mask_tensor = torch.from_numpy(input_cloth_mask).unsqueeze(0)
        cloth_image = Image.fromarray(make_image_array(gmm_outputs["cloth"]))
        cloth_mask_image = Image.fromarray(make_image_array(gmm_outputs["cloth_mask"])).convert('L')
        cloth_tensor = self.transform(cloth_image)
        cloth_mask_tensor = torch.from_numpy((np.array(cloth_mask_image) >= 128).astype(np.float32))
        cloth_mask_tensor.unsqueeze_(0)
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
        parse_mask_array = (person_mask_array > 0).astype(np.float32)

        # downsample shape
        parse_mask_orig = Image.fromarray((parse_mask_array * 255).astype(np.uint8))
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

            px = pose_array[i, 0]
            py = pose_array[i, 1]

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
        print(parse_mask_tensor.shape)
        print(image_head.shape)
        print(pose_map.shape)
        agnostic = torch.cat([parse_mask_tensor, image_head, pose_map], 0)

        parse_cloth_tensor.unsqueeze_(0)
        Image.fromarray(make_image_array(image_head)).save("../output/test/image_head.jpg")

        return {
            "agnostic": agnostic,
            "cloth": cloth_tensor,
            "cloth_mask": cloth_mask_tensor,
        }

    def predict(self, inputs):
        # TOM
        with torch.no_grad():
            self.model.cuda()
            self.model.eval()

            agnostic = inputs["agnostic"].unsqueeze(0).cuda()
            cloth_tensor = inputs["cloth"].unsqueeze(0).cuda()
            cloth_mask_tensor = inputs["cloth_mask"].unsqueeze(0).cuda()

            print("cloth", cloth_tensor.shape)
            print("cloth_mask", cloth_mask_tensor.shape)

            outputs = self.model(torch.cat([agnostic, cloth_tensor, cloth_mask_tensor], 1))
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            p_tryon = cloth_tensor * m_composite + p_rendered * (1 - m_composite)

            tryon_image = Image.fromarray(make_image_array(p_tryon))
            tryon_image.save("../output/test/tryon.png")
            Image.fromarray(make_image_array(m_composite)).save("../output/test/composite.jpg")
            Image.fromarray(make_image_array(cloth_tensor)).save("../output/test/warpcloth.jpg")
            Image.fromarray(make_image_array(p_rendered)).save("../output/test/rendered.jpg")
            return tryon_image


def load_cloth_image(outfit_id):
    # TODO: load images from a db service instead
    cloth_image = Image.open(os.path.join(DATA_DIR, f"{outfit_id}.jpg"))
    cloth_mask_image = Image.open(os.path.join(DATA_DIR, f"{outfit_id}_mask.jpg")).convert("L")
    return cloth_image, cloth_mask_image


def load_parse_image():
    """TODO: Replace with code that runs JPPNet
    """
    return Image.open(os.path.join(DATA_DIR, "person_parse.png")).convert("L")


def create_mask(parse_image):
    """TODO: Replace with code that runs JPPNet
    """
    img = np.array(parse_image)
    img[img > 0] = 255
    return Image.fromarray(img).convert("L")


class InferenceEngine:
    def __init__(self, gmm_model_path, tom_model_path):
        self.gmm = GMMEngine(gmm_model_path)
        self.tom = TOMEngine(tom_model_path)

    def load(self):
        self.gmm.load()
        self.tom.load()

    def infer(self, image_file, pose_data, person_parse_image, outfit_id="cloth"):
        # load image
        try:
            person_image = Image.open(image_file).convert("RGB")
        except Exception:
            raise Exception("Invalid image input data")

        pose_data_dict = json.load(pose_data)

        # get cloth image and mask using the outfit_id
        cloth_image, cloth_mask_image = load_cloth_image(outfit_id)
        # run person segmentation
        # person_parse_image = load_parse_image()
        # get binary body mask
        person_mask_image = create_mask(person_parse_image)
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
        tom_inputs = self.tom.prepare_inputs(
            gmm_out, person_image, person_parse_image, person_mask_image, pose_data_dict
        )
        # return try-on image
        output = self.tom.predict(tom_inputs)
        return output
