#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Depends,
    HTTPException,
    status,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
import numpy as np
import secrets
import os
from starlette.responses import FileResponse
from vton_service import InferenceEngine as VTON
import human_parsing


app = FastAPI(
    version="0.1",
    title="Virtual Trial Room Inference Server",
    description="API for performing virtual try-on using CP-VTON+",
)

security = HTTPBasic()

gmm_model_path = "../checkpoints/GMM/gmm_final.pth"
tom_model_path = "../checkpoints/TOM/tom_final.pth"
human_parsing_model = "../lip_jppnet_384.pb"
vton = VTON(gmm_model_path, tom_model_path)
vton.load()

parsing = human_parsing.load_model(human_parsing_model)


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username, os.getenv("API_USERNAME")
    )

    correct_password = secrets.compare_digest(
        credentials.password, os.getenv("API_PASSWORD")
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.post("/tryon")
async def virtual_tryon(
    image: UploadFile = File(...),
    posedata: UploadFile = File(...),
    username: str = Depends(get_current_username),
):
    # receive image and pose json file
    # TODO check if valid
    result = vton.infer(image.file, posedata.file)
    # return output image
    result.save("/tmp/tryon.jpg")

    return FileResponse("/tmp/tryon.jpg", media_type="image/jpeg")


@app.post("/parse")
async def run_parsing(
    image_file: UploadFile = File(...), username: str = Depends(get_current_username)
):
    image = np.array(Image.open(image_file.file).convert("RGB"))
    image = image[:, :, -1::-1]

    seg_map = human_parsing.predict(parsing, image)
    seg_color = human_parsing.colorize_segmentation(seg_map)

    Image.fromarray(seg_color).save("/tmp/segmap.jpg")
    return FileResponse("/tmp/segmap.jpg", media_type="image/jpeg")
