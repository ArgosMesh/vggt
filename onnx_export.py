# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)
dummy_input = torch.randn(4, 3, 224, 224)
#torch.onnx.export(model, dummy_input, "vggt.onnx", verbose = True, dynamo=True, report=True, opset_version=17,
#                    input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}})
torch.onnx.export(model, dummy_input, "vggt.onnx", verbose = True, dynamo=True, report=True, opset_version=17,
                    input_names=["input"], output_names=["output"] )