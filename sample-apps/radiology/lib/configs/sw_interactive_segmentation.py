# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import BasicUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

from monai.networks.nets.dynunet import DynUNet

# from sw_interactive_segmentation.api import (
#     get_network,
# )

logger = logging.getLogger(__name__)


class SWInteractiveSegmentationConfig(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = [
            "tumor",
            "background",
        ]
        
        self.label_names = {label:self.labels.index(label) for label in self.labels}
        print(self.label_names)
        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # Download PreTrained Model
        # Model is pretrained on PET scans from the AutoPET dataset
        if strtobool(self.conf.get("use_pretrained_model", "true")):
            url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}"
            url = f"{url}/sw_interactive_segmentation.pt"
            download_file(url, self.path[0])

        # Network
        in_channels = 1 + len(self.labels)
        self.network = DynUNet(
            spatial_dims=3,
            # 1 dim for the image, the other ones for the signal per label with is the size of image
            in_channels=in_channels,
            out_channels=len(self.labels),
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, [2, 2, 1]],
            upsample_kernel_size=[2, 2, 2, 2, [2, 2, 1]],
            norm_name="instance",
            deep_supervision=False,
            res_block=True,
        )

        self.target_spacing = [2.03642011, 2.03642011, 3.0]  # AutoPET default
        # Setting ROI size
        self.sw_roi_size = (128, 128, 128)



    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        return {
            self.name: lib.infers.SWInteractiveSegmentationInfer(
            path=self.path,
            network=self.network,
            labels=self.labels,
            label_names=self.label_names, 
            preload=strtobool(self.conf.get("preload", "false")),
            config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
            ),
            f"{self.name}_seg": lib.infers.SWInteractiveSegmentationInfer(
            path=self.path,
            network=self.network,
            labels=self.labels,
            label_names=self.label_names, 
            preload=strtobool(self.conf.get("preload", "false")),
            config={"cache_transforms": True, "cache_transforms_in_memory": True, "cache_transforms_ttl": 300},
            ),
        }
        # return task

    def trainer(self) -> Optional[TrainTask]:
        return None
