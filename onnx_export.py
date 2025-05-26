# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import glob
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import gradio as gr
import numpy as np
import torch

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import predictions_to_glb

device = "cpu"

print("Initializing and loading VGGT model...")

model = VGGT().float()
model.eval()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)
dummy_input = torch.randn(1, 4, 3, 224, 224, dtype=torch.float32)
torch.onnx.export(
    model,
    (dummy_input,),
    "vggt.onnx",
    verbose=True,
    dynamo=True,
    report=True,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
)
