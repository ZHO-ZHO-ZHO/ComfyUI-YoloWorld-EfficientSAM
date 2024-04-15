from .YOLO_WORLD_EfficientSAM import *
from collections import namedtuple
from PIL import Image

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped


crop_tensor4 = crop_ndarray4


def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)


def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results


def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iter)

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]


def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)


def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def inference_bbox(yolo_world_model, categories, iou_threshold, with_class_agnostic_nms, image, confidence):
    img = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    yolo_world_model.set_classes(categories)

    results = yolo_world_model.infer(img, confidence=confidence)
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(class_agnostic=with_class_agnostic_nms, threshold=iou_threshold)

    bboxes = detections.xyxy
    cv2_image = np.array(img)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    else:
        # Handle the grayscale image here
        # For example, you might want to convert it to a 3-channel grayscale image for consistency:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(detections.data['class_name'][i])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(detections.confidence[i])

    return results


def inference_segm(yolo_world_model, esam_model, categories, iou_threshold, with_class_agnostic_nms, image, confidence):
    img = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    yolo_world_model.set_classes(categories)
    results = yolo_world_model.infer(img, confidence=confidence)
    detections = sv.Detections.from_inference(results)
    detections = detections.with_nms(class_agnostic=with_class_agnostic_nms, threshold=iou_threshold)
    segms = inference_with_boxes(
        image=img,
        xyxy=detections.xyxy,
        model=esam_model,
        device=DEVICE
    )

    bboxes = detections.xyxy
    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(detections.data['class_name'][i])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
        scaled_mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(img.shape[0], img.shape[1]), mode='bilinear', align_corners=False)
        scaled_mask = scaled_mask.squeeze().squeeze()

        results[2].append(scaled_mask.numpy())
        results[3].append(detections.confidence[i])

    return results


class YoloworldBboxDetector:
    def __init__(self, yolo_world_model, categories, iou_threshold, with_class_agnostic_nms):
        self.yolo_world_model = yolo_world_model
        self.categories = process_categories(categories)
        self.iou_threshold = iou_threshold
        self.with_class_agnostic_nms = with_class_agnostic_nms

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None, esam_model=None):
        drop_size = max(drop_size, 1)
        if esam_model is None:
            detected_results = inference_bbox(self.yolo_world_model, self.categories, self.iou_threshold, self.with_class_agnostic_nms, image, threshold)
        else:
            detected_results = inference_segm(self.yolo_world_model, esam_model, self.categories, self.iou_threshold, self.with_class_agnostic_nms, image, threshold)

        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.yolo_world_model, self.categories, self.iou_threshold, self.with_class_agnostic_nms, image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)


class YoloworldSegmDetector:
    def __init__(self, bbox_detector, esam_model):
        self.bbox_detector = bbox_detector
        self.esam_model = esam_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        return self.bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook=detailer_hook, esam_model=self.esam_model)

    def detect_combined(self, image, threshold, dilation):
        bb = self.bbox_detector
        detected_results = inference_segm(bb.yolo_world_model, self.esam_model, bb.categories, bb.iou_threshold, bb.with_class_agnostic_nms, image, threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)


class Yoloworld_ESAM_DetectorProvider_Zho:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "yolo_world_model": ("YOLOWORLDMODEL",),
                "categories": ("STRING", {"default": "", "placeholder": "Please enter the objects to be detected separated by commas.", "multiline": True}),
                "iou_threshold": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01}),
                "with_class_agnostic_nms": ("BOOLEAN", {"default": False}),
            },

            "optional": {
                "esam_model_opt": ("ESAMMODEL",),
            }
        }

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"

    CATEGORY = "🔎YOLOWORLD_ESAM"

    def doit(self, yolo_world_model, categories, iou_threshold, with_class_agnostic_nms, esam_model_opt=None):
        bbox_detector = YoloworldBboxDetector(yolo_world_model, categories, iou_threshold, with_class_agnostic_nms)
        if esam_model_opt is not None:
            segm_detector = YoloworldSegmDetector(bbox_detector, esam_model_opt)
        else:
            segm_detector = None

        return bbox_detector, segm_detector


NODE_CLASS_MAPPINGS = {
    "Yoloworld_ESAM_DetectorProvider_Zho": Yoloworld_ESAM_DetectorProvider_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yoloworld_ESAM_DetectorProvider_Zho": "🔎Yoloworld ESAM Detector Provider",
}
