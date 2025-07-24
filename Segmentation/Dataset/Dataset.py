import os
import torch
import random
import json
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from Util import random_crop_around_lesion
import torchio as tio

class StrokeAI(Dataset):
    def __init__(self, 
                 CT_root, DWI_root, ADC_root, label_root, MRI_type,
                 mode='train',
                 map_file=None,
                 bounding_box=False,
                 transform=None,
                 padding=False,
                 slicing=False,
                 instance_normalize=False,
                 crop=False,
                 random_crop_ratio=0.0,
                 RotatingResize=False):
        
        self.ct_dir = CT_root
        self.dwi_dir = DWI_root
        self.adc_dir = ADC_root
        self.label_dir = label_root
        self.MRI_type = MRI_type
        self.mode = mode

        self.bounding = bounding_box
        self.transform = transform
        self.padding = padding
        self.slicing = slicing
        self.instance_normalize = instance_normalize
        self.cropping = crop
        self.random_crop_ratio = random_crop_ratio
        self.RotatingResize = RotatingResize

        random.seed(42)
        self.train_ids, self.test_ids = self.get_unique_ids()

        if map_file is not None:
            with open(map_file, 'r') as f:
                self.ct_map_mri = json.load(f)
    def get_unique_ids(self):
        # Extract unique IDs from filenames in CT directory
        ids = ['_'.join(fname.split('_')[:2]) for fname in os.listdir(self.ct_dir)]
        ids = list(set(ids))
        ids.sort()
        split_index = int(0.8 * len(ids))
        return ids[:split_index], ids[split_index:]

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        elif self.mode == 'test':
            return len(self.test_ids)
        else:
            raise ValueError("Mode must be 'train' or 'test'")
    def __getitem__(self, idx):
        if self.mode == 'train':
            unique_id = self.train_ids[idx]
        elif self.mode == 'test':
            unique_id = self.test_ids[idx]
        else:
            raise ValueError("Invalid mode")

        # Build file paths
        ct_path = os.path.join(self.ct_dir, f"{unique_id}_ct.nii.gz")
        if self.MRI_type == 'DWI':
            mri_path = os.path.join(self.dwi_dir, f"{unique_id}/{unique_id}_registered_mri.nii.gz")
        elif self.MRI_type == 'ADC':
            mri_path = os.path.join(self.adc_dir, f"{unique_id}_ADC_coregis.nii.gz")
        else:
            raise ValueError("Invalid MRI_type (must be 'ADC' or 'DWI')")

        label_path = os.path.join(self.label_dir, f"{unique_id}_registered_label.nii.gz")

        # Load images via SimpleITK
        ct_sitk = sitk.ReadImage(ct_path)
        mri_sitk = sitk.ReadImage(mri_path)
        label_sitk = sitk.ReadImage(label_path)

        # Process images
        if self.mode == 'train':
            ct_tensor, mri_tensor, label_tensor = self.preprocess(ct_sitk, mri_sitk, label_sitk)
        else:  # test mode
            ct_np = sitk.GetArrayFromImage(ct_sitk)
            mri_np = sitk.GetArrayFromImage(mri_sitk)
            label_np = sitk.GetArrayFromImage(label_sitk).astype(np.float32)

            ct_tensor = torch.from_numpy(ct_np).unsqueeze(0).float()
            mri_tensor = torch.from_numpy(mri_np).unsqueeze(0).float()
            label_tensor = torch.from_numpy(label_np).unsqueeze(0).float()

        return {'ct': ct_tensor, 'mri': mri_tensor, 'label': label_tensor}
    def preprocess(self, ct_sitk, mri_sitk, label_sitk):
        ct = sitk.GetArrayFromImage(ct_sitk)
        mri = sitk.GetArrayFromImage(mri_sitk)
        label = sitk.GetArrayFromImage(label_sitk)

        # Optional: pad off edge
        if self.padding:
            ct = ct[:-1, :-1, :-1]
            mri = mri[:-1, :-1, :-1]
            label = label[:-1, :-1, :-1]

        # Optional: slice along z
        if self.slicing:
            ct = ct[:, :, 90:(90 + self.slicing)]
            mri = mri[:, :, 90:(90 + self.slicing)]
            label = label[:, :, 90:(90 + self.slicing)]

        # Add channel dimension
        ct = np.expand_dims(ct, axis=0)
        mri = np.expand_dims(mri, axis=0)
        label = np.expand_dims(label, axis=0)

        # Optional: TorchIO-based random rotation + resize
        if self.RotatingResize:
            rot = tio.Affine(
                scales=np.random.uniform(0.8, 1.2, 3),
                degrees=np.random.uniform(-8, 8, 3),
                translation=(0, 0, 0),
                image_interpolation='linear'
            )
            label_affine = tio.Affine(
                scales=rot.scales,
                degrees=rot.degrees,
                translation=(0, 0, 0),
                image_interpolation='nearest'
            )

            subject_img = tio.Subject(ct=tio.ScalarImage(tensor=ct), mri=tio.ScalarImage(tensor=mri))
            subject_lbl = tio.Subject(label=tio.ScalarImage(tensor=label))

            ct = rot(subject_img).ct.numpy()
            mri = rot(subject_img).mri.numpy()
            label = label_affine(subject_lbl).label.numpy()

        # Optional: crop around lesion center
        if self.cropping:
            lesion_voxels = np.argwhere(label == 1)
            if len(lesion_voxels) == 0:
                raise ValueError("No lesion found")
            center = tuple(random.choice(lesion_voxels))
            ct, mri, label = random_crop_around_lesion(ct, mri, label, center, crop_size=(56, 56, 56), random_crop_prob=self.random_crop_ratio)

        # Convert to tensors
        ct_tensor = torch.tensor(ct).float()
        mri_tensor = torch.tensor(mri).float()
        label_tensor = torch.tensor(label).float()

        if self.instance_normalize:
            norm = torch.nn.InstanceNorm3d(1, affine=False)
            ct_tensor = norm(ct_tensor.unsqueeze(0)).squeeze(0).detach()
            mri_tensor = norm(mri_tensor.unsqueeze(0)).squeeze(0).detach()

        return ct_tensor, mri_tensor, label_tensor