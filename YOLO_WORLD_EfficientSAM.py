from typing import List
import folder_paths
import os
import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from inference.models import YOLOWorld

from .utils.efficient_sam import load, inference_with_boxes
from .utils.video import generate_file_name, calculate_end_frame_index, create_directory

current_directory = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

folder_paths.folder_names_and_paths["yolo_world"] = ([os.path.join(folder_paths.models_dir, "yolo_world")], folder_paths.supported_pt_extensions)

def process_categories(categories: str) -> List[str]:
    return [category.strip() for category in categories.split(',')]

def annotate_image(
    input_image: np.ndarray,
    detections: sv.Detections,
    categories: List[str],
    with_confidence: bool = False,
    thickness: int = 2,
    text_thickness: int = 2,
    text_scale: float = 1.0,
) -> np.ndarray:
    labels = [
        (
            f"{categories[class_id]}: {confidence:.3f}"
            if with_confidence
            else f"{categories[class_id]}"
        )
        for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=thickness)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image


class Yoloworld_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_world_model": (["yolo_world/l", "yolo_world/m", "yolo_world/s"], ),
            }
        }

    RETURN_TYPES = ("YOLOWORLDMODEL",)
    RETURN_NAMES = ("yolo_world_model",)
    FUNCTION = "load_yolo_world_model"
    CATEGORY = "🔎YOLOWORLD_ESAM"
  
    def load_yolo_world_model(self, yolo_world_model):
        YOLO_WORLD_MODEL = YOLOWorld(model_id=yolo_world_model)

        return [YOLO_WORLD_MODEL]
        

class ESAM_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["CUDA", "CPU"], ),
            }
        }

    RETURN_TYPES = ("ESAMMODEL",)
    RETURN_NAMES = ("esam_model",)
    FUNCTION = "load_esam_model"
    CATEGORY = "🔎YOLOWORLD_ESAM"
  
    def load_esam_model(self, device):
        if device == "CUDA":
            model_path = os.path.join(current_directory, "efficient_sam_s_gpu.jit")
        else:
            model_path = os.path.join(current_directory, "efficient_sam_s_cpu.jit")
            
        EFFICIENT_SAM_MODEL = torch.jit.load(model_path)

        return [EFFICIENT_SAM_MODEL]

class Yoloworld_ESAM_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolo_world_model": ("YOLOWORLDMODEL",),
                "esam_model": ("ESAMMODEL",),
                "image": ("IMAGE",),
                "categories": ("STRING", {"default": "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step":0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step":0.01}),
                "box_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_thickness": ("INT", {"default": 2, "min": 1, "max": 5}),
                "text_scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step":0.01}),
                "with_confidence": ("BOOLEAN", {"default": True}),
                "with_class_agnostic_nms": ("BOOLEAN", {"default": False}),
                "with_segmentation": ("BOOLEAN", {"default": True}),
                "mask_combined": ("BOOLEAN", {"default": True}),
                "mask_extracted": ("BOOLEAN", {"default": True}),
                "mask_extracted_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "yoloworld_esam_image"
    CATEGORY = "🔎YOLOWORLD_ESAM"
                       
    def yoloworld_esam_image(self, image, yolo_world_model, esam_model, categories, confidence_threshold, iou_threshold, box_thickness, text_thickness, text_scale, with_segmentation, mask_combined, with_confidence, with_class_agnostic_nms, mask_extracted, mask_extracted_index):
        categories = process_categories(categories)
        processed_images = []
        processed_masks = []
        for img in image:
            img = np.clip(255. * img.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)          
            YOLO_WORLD_MODEL = yolo_world_model
            YOLO_WORLD_MODEL.set_classes(categories)
            results = YOLO_WORLD_MODEL.infer(img, confidence=confidence_threshold)
            detections = sv.Detections.from_inference(results)
            detections = detections.with_nms(
                class_agnostic=with_class_agnostic_nms,
                threshold=iou_threshold
            )

            combined_mask = None
            if with_segmentation:
                detections.mask = inference_with_boxes(
                    image=img,
                    xyxy=detections.xyxy,
                    model=esam_model,
                    device=DEVICE
                )
                if mask_combined:
                    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    det_mask = detections.mask
                    for mask in det_mask:
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
                    masks_tensor = torch.tensor(combined_mask, dtype=torch.float32)
                    processed_masks.append(masks_tensor) 
                else:
                    det_mask = detections.mask
                    
                    if mask_extracted:
                        mask_index = mask_extracted_index
                        selected_mask = det_mask[mask_index]
                        masks_tensor = torch.tensor(selected_mask, dtype=torch.float32)
                    else:
                        masks_tensor = torch.tensor(det_mask, dtype=torch.float32)
                        
                    processed_masks.append(masks_tensor)  
                
            output_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            output_image = annotate_image(
                input_image=output_image,
                detections=detections,
                categories=categories,
                with_confidence=with_confidence,
                thickness=box_thickness,
                text_thickness=text_thickness,
                text_scale=text_scale,
            )
            
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            output_image = torch.from_numpy(output_image.astype(np.float32) / 255.0).unsqueeze(0)
            
            processed_images.append(output_image)

        new_ims = torch.cat(processed_images, dim=0)
        
        if processed_masks:
            new_masks = torch.stack(processed_masks, dim=0)
        else:
            new_masks = torch.empty(0)

        return new_ims, new_masks


NODE_CLASS_MAPPINGS = {
    "Yoloworld_ModelLoader_Zho": Yoloworld_ModelLoader_Zho,
    "ESAM_ModelLoader_Zho": ESAM_ModelLoader_Zho,
    "Yoloworld_ESAM_Zho": Yoloworld_ESAM_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yoloworld_ModelLoader_Zho": "🔎Yoloworld Model Loader",
    "ESAM_ModelLoader_Zho": "🔎ESAM Model Loader",
    "Yoloworld_ESAM_Zho": "🔎Yoloworld ESAM",
}
