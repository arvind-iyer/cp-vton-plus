#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import (FastAPI, Form, File, UploadFile, Header, Depends,
                     HTTPException, status)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import FileResponse

app = FastAPI(version="0.1",
              title="Virtual Trial Room Inference Server",
              description="API for performing virtual try-on using CP-VTON+")

security = HTTPBasic()


def get_current_username(
        credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username,
                                              os.getenv("API_USERNAME"))

    correct_password = secrets.compare_digest(credentials.password,
                                              os.getenv("API_PASSWORD"))

    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            details="Incorrect username or password",
                            headers={"WWW-Authenticate": "Basic"})


@app.post("/tryon")
def virtual_tryon(image: UploadFile = File(...),
                  posedata: UploadFile = File(...),
                  username: str = Depends(get_current_username)):
    # receive image and pose json file
    # check if valid
    # ensure GMM model is ready and loaded
    # run GMM
    # ensure TOM model is ready and loaded
    # run TOM
    # return output image

    pass
