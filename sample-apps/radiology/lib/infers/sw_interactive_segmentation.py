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

from typing import Callable, Sequence, Union, Dict, Any
import logging

import torch
from monai.apps.deepgrow.transforms import (
    AddGuidanceFromPointsd,
    AddGuidanceSignald,
    Fetch2DSliced,
    ResizeGuidanced,
    RestoreLabeld,
    SpatialCropGuidanced,
)
from monailabel.transform.post import Restored
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsChannelFirstd,
    AsChannelLastd,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Resized,
    Spacingd,
    ToNumpyd,
    SqueezeDimd,
    MapTransform,
)
from monai.transforms.transform import MapTransform
from monai.data import decollate_batch

from monailabel.interfaces.utils.transform import run_transforms
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from sw_interactive_segmentation.api import (
    get_pre_transforms, 
    get_post_transforms,
    get_inferers,
    get_pre_transforms_val_as_list_monailabel,
)
from sw_interactive_segmentation.utils.helper import AttributeDict
from sw_interactive_segmentation.utils.transforms import AddGuidanceSignal, PrintDatad

from monai.utils import set_determinism

logger = logging.getLogger(__name__)

class SWInteractiveSegmentationInfer(BasicInferTask):

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        label_names=None,
        dimension=3,
        description="",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.label_names = label_names

        self.args = AttributeDict()
        self.args.no_log = True
        self.args.output = None
        self.args.dataset = "AutoPET"
        self.args.train_crop_size = (128,128,128)
        self.args.val_crop_size = None
        self.args.inferer = "SlidingWindowInferer"
        self.args.sw_roi_size = (128,128,128)
        self.args.train_sw_batch_size = 1
        self.args.val_sw_batch_size = 1
        self.args.debug = False
        set_determinism(42)


    def pre_transforms(self, data=None) -> Sequence[Callable]:
        print("#########################################")
        device = data.get("device") if data else None
        t_val = get_pre_transforms_val_as_list_monailabel(self.label_names, device, self.args, input_keys=["image"])
        #t_val = []
        #t_val.append(NoOpd())
        #t_val.append(LoadImaged(keys="image", reader="ITKReader"))

        #t_val.append(NoOpd())
        return t_val

    def inferer(self, data=None) -> Inferer:
        _, val_inferer = get_inferers(
            inferer=self.args.inferer,
            sw_roi_size=self.args.sw_roi_size,
            train_crop_size=self.args.train_crop_size,
            val_crop_size=self.args.val_crop_size,
            train_sw_batch_size=self.args.train_sw_batch_size,
            val_sw_batch_size=self.args.val_sw_batch_size,
        )
        return val_inferer

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        device = data.get("device") if data else None
        #return get_post_transforms(self.labels, device)
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            #RestoreLabeld(keys="pred", ref_image="image", mode="nearest"),
            #AsChannelLastd(keys="pred"),
            #Restored(keys="pred", ref_image="image"),
            #ToNumpyd(keys="pred"),
            PrintDatad(),
            EnsureTyped(keys="pred", device="cpu" if data else None, dtype=torch.uint8),
        ]


class NoOpd(MapTransform):
    def __init__(self, keys= None):
        """
        A transform which does nothing
        """
        super().__init__(keys)

    def __call__(
        self, data
        ):
        #print(data["image"])
        try:
            print(data["image"])
            print(data["image_path"])
        except AttributeError:
            pass
        print(type(data["image"]))
        return data

    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """

        inferer = self.inferer(data)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")

        network = self._get_network(device, data)
        if network:
            inputs = data[self.input_key]
            inputs = inputs if torch.is_tensor(inputs) else torch.from_numpy(inputs)
            inputs = inputs[None] if convert_to_batch else inputs
            inputs = inputs.to(torch.device(device))

            with torch.no_grad():
                outputs = inferer(inputs, network)

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            if convert_to_batch:
                if isinstance(outputs, dict):
                    outputs_d = decollate_batch(outputs)
                    outputs = outputs_d[0]
                else:
                    outputs = outputs[0]

            data[self.output_label_key] = outputs
        else:
            # consider them as callable transforms
            data = run_transforms(data, inferer, log_prefix="INF", log_name="Inferer")
        return data

