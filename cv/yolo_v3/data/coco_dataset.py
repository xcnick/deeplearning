import skimage
from pycocotools.coco import COCO


class SimpleCoCoDataset(Dataset):
    def __init__(self, rootdir, set_name="val2017", transform=None):
        self.rootdir, self.set_name = rootdir, set_name
        self.transform = transform
        self.coco = COCO(
            os.path.join(self.rootdir, "annotations", "instances_" + self.set_name + ".json")
        )
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely

        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c["id"]
            self.coco_labels_inverse[c["id"]] = len(self.classes)
            self.classes[c["name"]] = len(self.classes)

        # labels:              {new_index:  names}
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        ann = self.load_anns(index)
        sample = {"img": img, "ann": ann}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath = os.path.join(self.rootdir, "images", self.set_name, image_info["file_name"])

        img = skimage.io.imread(imgpath)
        return img.astype(np.float32) / 255.0

    def load_anns(self, index):
        annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        # anns is num_anns x 5, (x1, x2, y1, y2, new_idx)
        anns = np.zeros((0, 5))

        # skip the image without annoations
        if len(annotation_ids) == 0:
            return anns

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            # skip the annotations with width or height < 1
            if a["bbox"][2] < 1 or a["bbox"][3] < 1:
                continue

            ann = np.zeros((1, 5))
            ann[0, :4] = a["bbox"]
            ann[0, 4] = self.coco_labels_inverse[a["category_id"]]
            anns = np.append(anns, ann, axis=0)

        # (x1, y1, width, height) --> (x1, y1, x2, y2)
        anns[:, 2] += anns[:, 0]
        anns[:, 3] += anns[:, 1]

        return anns

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image["width"]) / float(image["height"])


class Normilizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
        self.std  = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    def __call__(self, sample):
        image, anns = sample['img'], sample['ann']
        return {'img':(image.astype(np.float32)-self.mean)/ self.std,
                'ann':anns}