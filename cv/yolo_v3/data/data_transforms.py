import random
import numpy as np
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Resize(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def resize_boxes(self, boxes, original_size, new_size):
        # type: (Tensor, List[int], List[int]) -> Tensor
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)

    def __call__(self, image, target):
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        max_size = float(torch.max(im_shape))
        scale_factor = self.image_size / max_size
        image = torch.nn.functional.interpolate(
            image[None],
            size=[round(h * scale_factor), round(w * scale_factor)],
            mode="bilinear",
            align_corners=True,
        )[0]

        height, width = image.shape[-2:]
        dim_diff = np.abs(height - width)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = (0, 0, pad1, pad2) if height <= width else (pad1, pad2, 0, 0)
        image = torch.nn.functional.pad(image, pad, "constant", value=0)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = self.resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target


def collater(data):
    images = torch.stack([one[0] for one in data])
    boxes = [one[1]["boxes"] for one in data]
    labels = [one[1]["labels"] for one in data]

    max_num_boxes = max(label.shape[0] for label in labels)

    boxes_padded = torch.zeros((len(labels), max_num_boxes, 5))

    for idx, (label, box) in enumerate(zip(labels, boxes)):
        if label.shape[0] > 0:
            boxes_padded[idx, : label.shape[0], :] = torch.cat((label.unsqueeze(1), box), 1)

    targets = boxes_padded

    return images, targets
