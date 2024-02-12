import os
import torch
import numpy as np
import math
import cv2
import json
import copy
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import gaussian_filter
from skimage import exposure


from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
from .reactor_swapper import swap_face, get_current_faces_model, analyze_faces, get_face_single
from .reactor_logger import logger

import folder_paths
from nodes import MAX_RESOLUTION

import comfy
from .utils import tensor_to_pil, pil_to_tensor, tensor2pil, pil2tensor, pil2mask
from .modules import shared
from .modules.scripts import USDUMode, USDUSFMode, Script
from .modules.processing import StableDiffusionProcessing
from .modules.upscaler import UpscalerData

# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    return [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True})),
        ("tiled_decode", ("BOOLEAN", {"default": False})),
    ]


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return prepare_inputs(USDU_base_inputs())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "HD Nodes/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers = [None]
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode
        )

        #
        # Running the script
        #
        script = Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting images
        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


class UltimateSDUpscaleNoUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "HD Nodes/upscaling"

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode):

        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = None
        shared.batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(upscaled_image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1, force_uniform_tiles, tiled_decode
        )

        script = Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1)

        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)



class MaskCombineOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "index": ("INT", {"default": 0, "min": 0, "max": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine"

    CATEGORY = "HD Nodes"

    def combine(self, mask1, mask2, index=0):
        
        if (index == 0):
            result = mask1
        else:
            result = mask2

        return (result,)

class MaskCoverOp:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("MASK",),
                "source": ("MASK",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "cover"

    CATEGORY = "HD Nodes"

    def cover(self, destination, source, x, y):
        output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
        source = source.reshape((-1, source.shape[-2], source.shape[-1]))

        left, top = (x, y,)
        right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
        visible_width, visible_height = (right - left, bottom - top,)

        source_portion = source[:, :visible_height, :visible_width]
        destination_portion = destination[:, top:bottom, left:right]

        output[:, top:bottom, left:right] = destination_portion * source_portion
        
        output_ones = output.norm(1)
        destination_ones = destination_portion.norm(1)
        if (output_ones / destination_ones) < 0.2:
            result = 1
        else:
            result = 0

        return (result,)

class GetFaceIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "getindex"

    CATEGORY = "HD Nodes"

    def getindex(self, input_image):
        target_img = tensor_to_pil(input_image)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        target_faces = analyze_faces(target_img)
        target_face, wrong_gender = get_face_single(target_img, target_faces)
        if target_face is not None:
            result = 1
        else:
            result = 0
        return (result,)


class GetMaskArea:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask": ("MASK",),
                "h_offset": ("INT", {"default": 100, "step":10}),
                "h_cutoff": ("FLOAT", {"default": 0, "step":0.001}),
                "max_width": ("INT", {"default": 1600, "min": 0, "max": MAX_RESOLUTION, "step": 100}),
                "min_height": ("INT", {"default": 2400, "min": 0, "max": MAX_RESOLUTION, "step": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    FUNCTION = "getimage"

    CATEGORY = "HD Nodes"

    def get_mask_aabb(self, masks):
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

        b = masks.shape[0]

        bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
        is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
        for i in range(b):
            mask = masks[i]
            if mask.numel() == 0:
                continue
            if torch.max(mask != 0) == False:
                is_empty[i] = True
                continue
            y, x = torch.where(mask)
            bounding_boxes[i, 0] = torch.min(x)
            bounding_boxes[i, 1] = torch.min(y)
            bounding_boxes[i, 2] = torch.max(x)
            bounding_boxes[i, 3] = torch.max(y)

        return bounding_boxes, is_empty

    def getimage(self, image1, image2, mask, h_offset = 100, h_cutoff=0, max_width=1600, min_height=2400):
        bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
        boxes, is_empty = self.get_mask_aabb(bounds)

        padding = 0.02
        box = boxes[0]
        H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
        hh = int(int(H.item()) * (1.0 - h_cutoff))
        yy = int(Y.item()) - h_offset
        hh = int((hh + 50) * (1 + padding))
        ww = int(int(W.item()) * (1 + 20 * padding))
        #X = int(X.item()) - 10
        #W = int(W.item()) + 20
        xx = max(int(int(X.item()) - ww / ( 1 + 20 * padding) * 10 * padding), 0)
        image1_copy = copy.deepcopy(image1)
        image11 = image1_copy[:,yy:yy+hh,xx:xx+ww]
        image2_copy = copy.deepcopy(image2)
        image22 = image2_copy[:,yy:yy+hh,xx:xx+ww]
        mask = mask[:,yy:yy+hh,xx:xx+ww]
        return (image11, image22, mask)

# IMAGE LEVELS NODE

class HD_Image_Levels:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 0.1}),
                "mid_level": ("FLOAT", {"default": 127.5, "min": 0.0, "max": 255.0, "step": 0.1}),
                "white_level": ("FLOAT", {"default": 255, "min": 0.0, "max": 255.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_image_levels"

    CATEGORY = "HD Nodes/Image/Adjustment"

    def apply_image_levels(self, image, black_level, mid_level, white_level):
        tensor_images = []
        for timg in image:
            img = tensor2pil(timg)
            levels = self.AdjustLevels(black_level, mid_level, white_level)
            tensor_images.append(pil2tensor(levels.adjust(img)))
        tensor_images = torch.cat(tensor_images, dim=0)

        # Return adjust image tensor
        return (tensor_images, )


    class AdjustLevels:
        def __init__(self, min_level, mid_level, max_level):
            self.min_level = min_level
            self.mid_level = mid_level
            self.max_level = max_level

        def isBright(self, pil_image):
            gray_img = np.array(pil_image.convert('L'))
            
            # 获取形状以及长宽
            img_shape = gray_img.shape
            height, width = img_shape[0], img_shape[1]
            size = gray_img.size
            # 灰度图的直方图
            hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            
            # 计算灰度图像素点偏离均值(128)程序
            a = 0
            ma = 0
            reduce_matrix = np.full((height, width), 128)
            shift_value = gray_img - reduce_matrix
            shift_sum = sum(map(sum, shift_value))

            da = shift_sum / size
            # 计算偏离128的平均偏差
            for i in range(256):
                ma += (abs(i-128-da) * hist[i])
            m = abs(ma / size)
            # 亮度系数
            k = abs(da) / m
            # print(k)
            if k[0] > 1:
                # 过亮
                if da > 0:
                    #print("过亮")
                    return True
                else:
                    #print("过暗")
                    return False
            else:
                #print("亮度正常")
                return False

        def adjust(self, im):

            if not self.isBright(im):
                im_arr = np.array(im)
                im_arr[im_arr < self.min_level] = self.min_level
                im_arr = (im_arr - self.min_level) * \
                    (255 / (self.max_level - self.min_level))
                im_arr[im_arr < 0] = 0
                im_arr[im_arr > 255] = 255
                im_arr = im_arr.astype(np.uint8)
                
                im = Image.fromarray(im_arr)
                im = ImageOps.autocontrast(im, cutoff=self.max_level)

            return im

class SmoothEdge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "sigma": ("FLOAT", {"default":1.5, "min":0.0, "max":12.0, "step":0.1}),
                "gamma": ("INT", {"default": 20}),
            },
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "smooth"

    CATEGORY = "HD Nodes"

    def img2tensor(self, img, bgr2rgb=True, float32=True):

        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    def smooth_region(self, image, tolerance):
        from scipy.ndimage import gaussian_filter
        image = image.convert("L")
        mask_array = np.array(image)
        smoothed_array = gaussian_filter(mask_array, sigma=tolerance)
        threshold = np.max(smoothed_array) / 2
        smoothed_mask = np.where(smoothed_array >= threshold, 255, 0).astype(np.uint8)
        smoothed_mask = exposure.adjust_gamma(smoothed_mask, gamma=20)
        smoothed_image = Image.fromarray(smoothed_mask, mode="L")
        return ImageOps.invert(smoothed_image.convert("RGB"))

    def smooth(self, masks, sigma=128, gamma=20):
        if masks.ndim > 3:
            regions = []
            for mask in masks:
                mask_np = np.clip(255. * mask.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(mask_np, mode="L")
                region_mask = self.smooth_region(pil_image, sigma)
                region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
                regions.append(region_tensor)
            regions_tensor = torch.cat(regions, dim=0)
            return (regions_tensor,)
        else:
            mask_np = np.clip(255. * masks.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(mask_np, mode="L")
            region_mask = self.smooth_region(pil_image, sigma)
            region_tensor = pil2mask(region_mask).unsqueeze(0).unsqueeze(1)
            return (region_tensor,)

NODE_CLASS_MAPPINGS = {
    "Combine HDMasks": MaskCombineOp,
    "Cover HDMasks": MaskCoverOp,
    "HD FaceIndex": GetFaceIndex,
    "HD SmoothEdge": SmoothEdge,
    "HD GetMaskArea": GetMaskArea,
    "HD Image Levels": HD_Image_Levels,
    "HD UltimateSDUpscale": UltimateSDUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Combine HDMasks": "Combine HDMasks",
    "Cover HDMasks": "Cover HDMasks",
    "HD FaceIndex": "HD FaceIndex",
    "HD SmoothEdge": "HD SmoothEdge",
    "HD GetMaskArea": "HD GetMaskArea",
    "HD Image Levels": "HD Image Levels",
    "HD UltimateSDUpscale": "HD Ultimate SD Upscale",
}
