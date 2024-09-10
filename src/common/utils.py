################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import ctypes
import datetime
import os
from pathlib import Path
import sys


sys.path.append('/opt/nvidia/deepstream/deepstream/lib')


def long_to_uint64(l):
    value = ctypes.c_uint64(l & 0xffffffffffffffff).value
    return value


def get_file_name_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0]


def generate_filename(prefix=None, postfix=None, extension=None):
    filename = ""
    if prefix:
        filename = f"{prefix}_"

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename += f"{timestamp}"

    if postfix:
        filename += f"_{postfix}"

    if extension:
        filename += f".{extension}"

    return filename


def insert_folder_nth_level(path, folder, level):
    parts = path.split(os.path.sep)
    parts.insert(level, folder)
    return os.path.sep.join(parts)


def replace_model_engine_file_in_nvinfer_plugin(pgie, platform_info):
    """ Used when each model file support only specific GPU.
        Currently encountered in models we converted to TensorRT: YOLO-X, OWLV2.
        TODO: We need to learn this issue better:
            1. How to check if the model file is specific to GPU?
            2. How come the sample apps do not have this issue? How did they achieve it?
            3. Is the model file specific to GPU or GPU type (e.g. Tesla, Quadro, etc.)?
    """
    gpu_name = platform_info.get_gpu_name()
    model_engine_file = insert_folder_nth_level(pgie.get_property("model-engine-file"), gpu_name, -1)
    if not os.path.exists(model_engine_file):
        raise FileNotFoundError(f"Model engine file {model_engine_file} does not exist")

    print(f"Setting {model_engine_file=}\n")
    pgie.set_property("model-engine-file", model_engine_file)


def check_and_normalize_inputs(inputs):
    for i, uri in enumerate(inputs):
        if not uri.startswith("rtsp://") and not uri.startswith("file://") \
            and not uri.startswith("http://") and not uri.startswith("https://"):
            path = Path(uri).resolve(strict=True)
            if os.path.exists(path):
                inputs[i] = f"file://{path}"
            else:
                sys.stderr.write(f"File {i} does not found\n")
                sys.exit(1)

        elif uri.startswith("file://"):
            path = Path(uri[7:]).resolve(strict=True)
            if not os.path.exists(path):
                sys.stderr.write(f"File {i} does not found\n")
                sys.exit(1)

        else:
            # RTSP URI
            pass

    return inputs
