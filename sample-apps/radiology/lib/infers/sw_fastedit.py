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
import shutil
import json

import copy
import torch
import numpy as np
import nibabel as nib

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
    Compose
)
from monai.transforms.transform import MapTransform
from monai.data import decollate_batch

from monailabel.interfaces.utils.transform import run_transforms
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask, CallBackTypes
from sw_fastedit.api import (
    get_pre_transforms, 
    get_post_transforms,
    get_inferers,
    # get_pre_transforms_val_as_list_monailabel,
)
from sw_fastedit.utils.helper import AttributeDict
from sw_fastedit.transforms import AddGuidanceSignal, PrintDatad, AddEmptySignalChannels, NormalizeLabelsInDatasetd, SignalFillEmptyd

from monai.utils import set_determinism
from pathlib import Path
import os

from monai.transforms import (
    LoadImaged,
    Orientationd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    Identityd,
)

from monai.inferers import SimpleInferer, SlidingWindowInferer


logger = logging.getLogger(__name__)

class SWFastEdit(BasicInferTask):

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        labels=None,
        label_names=None,
        dimension=3,
        target_spacing=(1.0, 1.0, 1.0),
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
        self.target_spacing = target_spacing


        self.args = AttributeDict()
        # self.args.no_log = True
        # self.args.output = None
        # self.args.output_dir = None
        # self.args.dataset = "AutoPET"
        # self.args.train_crop_size = (128,128,128)
        # Either no crop with None or crop like (128,128,128)
        self.val_crop_size = None
        # self.args.debug = False
        self.path = '/projects/mhadlich_segmentation/data/monailabel'
        set_determinism(42)
        self.model_state_dict = "net"
        self.load_strict = True
        self._amp = True
        

        # Inferer parameters
        self.sw_overlap = 0.5
        self.sw_roi_size = (128,128,128)
        self.train_sw_batch_size = 8
        self.val_sw_batch_size = 1



    def pre_transforms(self, data=None) -> Sequence[Callable]:
        # print("#########################################")
        data['label_dict'] = self.label_names
        data['label_names'] = self.label_names

        cpu_device = torch.device("cpu")
        device = data.get("device") if data else None
        loglevel = logging.DEBUG
        input_keys=("image")

        
        t = []
        # t_val_1, t_val_2 = get_pre_transforms_val_as_list_monailabel(self.label_names, device, self.args, input_keys=["image"])
        t_val_1 = [
            # Initial transforms on the inputs done on the CPU which does not hurt since they are executed asynchronously and only once
            # InitLoggerd(loglevel=loglevel, no_log=True, log_dir=None),
            LoadImaged(keys=input_keys, reader="ITKReader", image_only=False),
            EnsureChannelFirstd(keys=input_keys),
            NormalizeLabelsInDatasetd(keys="label", labels=self.label_names, device=cpu_device),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.95, b_min=0.0, b_max=1.0, clip=True, relative=False
            ),
            EnsureTyped(keys=input_keys, device=device, data_type="tensor"),
        ]
        t.extend(t_val_1)
        self.add_cache_transform(t, data)
        t_val_2 = [
            AddEmptySignalChannels(keys=input_keys, device=device),
            AddGuidanceSignal(
                keys=input_keys,
                sigma=1,
                disks=True,
                device=device,
            ),
            Orientationd(keys=input_keys, axcodes="RAS"),
            Spacingd(keys=input_keys, pixdim=self.target_spacing),
            CenterSpatialCropd(keys=input_keys, roi_size=self.val_crop_size)
            if self.val_crop_size is not None
            else Identityd(keys=input_keys, allow_missing_keys=True),
            SignalFillEmptyd(input_keys),
            # DivisiblePadd(keys=input_keys, k=64, value=0) if args.inferer == "SimpleInferer" else Identityd(keys=input_keys, allow_missing_keys=True),
        ]
        t.extend(t_val_2)
        #t_val = []
        #t_val.append(NoOpd())
        #t_val.append(LoadImaged(keys="image", reader="ITKReader"))

        #t_val.append(NoOpd())
        return t

    def inferer(self, data=None) -> Inferer:
        sw_params = {
            "roi_size": self.sw_roi_size,
            "mode":"gaussian",
            "cache_roi_weight_map": False,
            "overlap": self.sw_overlap,
        }
        eval_inferer = SlidingWindowInferer(
            sw_batch_size=self.val_sw_batch_size,
            **sw_params
        )
        return eval_inferer

        # _, val_inferer = get_inferers(
        #     inferer=self.args.inferer,
        #     sw_roi_size=self.args.sw_roi_size,
        #     train_crop_size=self.args.train_crop_size,
        #     val_crop_size=self.args.val_crop_size,
        #     train_sw_batch_size=self.args.train_sw_batch_size,
        #     val_sw_batch_size=self.args.val_sw_batch_size,
        #     cache_roi_weight_map=False,
        # )
        # return val_inferer

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        device = data.get("device") if data else None
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            Restored(keys="pred", ref_image="image"),
            EnsureTyped(keys="pred", device="cpu" if data else None), #, dtype=torch.uint8),
        ]

    def run_inferer(self, data: Dict[str, Any], convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key stored that will be used for post-processing
        """
        logger.info("#################")
        inferer = self.inferer(data)
        logger.info(f"Inferer:: {device} => {inferer.__class__.__name__} => {inferer.__dict__}")
        
        data["path"] = self.path
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

    def __call__(self, request, callbacks= None):
        if callbacks is None:
            callbacks = {}
        callbacks[CallBackTypes.POST_TRANSFORMS] = post_callback
        
        return super().__call__(request, callbacks)

    # def run_invert_transforms(self, data: Dict[str, Any], pre_transforms, names):
    #     if names is None:
    #         return data

    #     pre_names = dict()
    #     transforms = []
    #     for t in reversed(pre_transforms):
    #         if hasattr(t, "inverse"):
    #             pre_names[t.__class__.__name__] = t
    #             transforms.append(t)

    #     # Run only selected/given
    #     if len(names) > 0:
    #         transforms = [pre_transforms[n if isinstance(n, str) else n.__name__] for n in names]


    #     d = run_transforms(data, transforms, inverse=True, log_prefix="INV")
    #     d = copy.deepcopy(dict(data))
    #     d[self.input_key] = data[self.output_label_key]
    #     d = run_transforms(d, transforms, inverse=True, log_prefix="INV")
    #     data[self.output_label_key] = d[self.input_key]
    #     return data


def post_callback(data): 
    path = '/projects/mhadlich_segmentation/data/monailabel'
    
    #for k,v in data.items():
    #    print(f"k: {k}, v: {v}")
    image_name = Path(os.path.basename(data["image_path"]))
    true_image_name = image_name.name.removesuffix(''.join(image_name.suffixes))
    image_folder = Path(data["image_path"]).parent
    print(f"{true_image_name=}")
    print(f"{image_folder}")

    # Save the clicks
    clicks_per_label = {}
    for key in data['label_dict'].keys():
        clicks_per_label[key] = data[key]
        assert isinstance(data[key], list)
    logger.info(f"Now dumping dict: {clicks_per_label} to file {path}/clicks.json ...")
    with open(f"{path}/clicks.json", "w") as clicks_file:
        json.dump(clicks_per_label, clicks_file)

    # Save debug NIFTI, not fully working since the inverse transform of the image is not avaible
    if False:
        logger.info("SAVING NIFTI")
        inputs = data["image"]
        pred = data["pred"]
        logger.info(f"inputs.shape is {inputs.shape}")
        logger.info(f"sum of fgg is {torch.sum(inputs[1])}")
        logger.info(f"sum of bgg is {torch.sum(inputs[2])}")
        logger.info(f"Image path is {data['image_path']}, copying file")
        shutil.copyfile(data['image_path'], f"{path}/im.nii.gz")
        #save_nifti(f"{path}/im", inputs[0].cpu().detach().numpy())
        save_nifti(
            f"{path}/guidance_fgg", inputs[1].cpu().detach().numpy()
        )
        save_nifti(
            f"{path}/guidance_bgg", inputs[2].cpu().detach().numpy()
        )
        logger.info(f"pred.shape is {pred.shape}")
        save_nifti(
            f"{path}/pred", pred.cpu().detach().numpy()
        )
    return data

def save_nifti(name, im):
    affine = np.eye(4)
    affine[0][0] = -1
    ni_img = nib.Nifti1Image(im, affine=affine)
    ni_img.header.get_xyzt_units()
    ni_img.to_filename(f"{name}.nii.gz")

