import os
import pickle
from itertools import compress
from tqdm import tqdm
import numpy as np
from PIL import Image
import requests
import zipfile
import json
from sklearn.model_selection import train_test_split
import torch
from deel.puncc.api.utils import hungarian_assignment
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import Dataset


# Coco Dataset class
class CocoDataset(Dataset):
    def __init__(self, shuffle=True):
        URL_TYPE = "coco_url"
        self.annotations_path = self.download_instances_val2017()
        # Load the annotations
        with open(self.annotations_path, "r") as f:
            coco_annotations = json.load(f)
            # list of dicts with the keys: ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
            self.image_infos = coco_annotations["images"]
            # list of dicts with the keys: ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
            self.image_annotations = coco_annotations["annotations"]
            # class classes
            self.classes = coco_annotations["categories"]
        # Extract the flickr urls from the annotations
        self.image_urls = [image[URL_TYPE] for image in self.image_infos]
        if shuffle:
            np.random.seed(42)
            np.random.shuffle(self.image_urls)  # shuffle the flickr urls

        # Create a mapping from category ids to category names
        self.categoryid2name = {
            category["id"]: category["name"] for category in self.classes
        }
        # Create a mapping from flickr urls to image ids
        self.url2imageid = {image[URL_TYPE]: image["id"] for image in self.image_infos}
        # Create a mapping from image ids to bboxes (annotations)
        self.imageid2bbox = {
            image_id: []
            for image_id in np.unique(
                [annotation["image_id"] for annotation in self.image_annotations]
            )
        }
        # Create a mapping from image ids to category names
        self.imageid2class = {
            image_id: []
            for image_id in np.unique(
                [annotation["image_id"] for annotation in self.image_annotations]
            )
        }

        # remove urls that do not have annotations
        self.image_urls = [
            url for url in self.image_urls if self.url2imageid[url] in self.imageid2bbox
        ]

        # Format the annotations
        self._prepare_annotations()

    def _prepare_annotations(self):
        for annotation in self.image_annotations:
            # Format the annotation and append it to the list of bboxes
            self.imageid2bbox[annotation["image_id"]].append(
                self._format_annotation(annotation["bbox"])
            )
            self.imageid2class[annotation["image_id"]].append(
                self.categoryid2name[annotation["category_id"]]
            )

    def _format_annotation(self, annotation):
        # Annotation is in the format [x, y, width, height]
        # Format it to be in the format [x1, y1, x2, y2]
        # where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right
        # corner of the bounding box
        annotation[2] += annotation[0]
        annotation[3] += annotation[1]
        return annotation

    def _download_image(self, url):
        # Download the image using the url
        # response = requests.get(url)
        # img_data = response.content
        # image = Image.open(BytesIO(img_data)) # Open the image using PIL
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def download_instances_val2017(self):
        # URL for the COCO 2017 validation annotations
        url = "https://raw.githubusercontent.com/deel-ai/puncc/refs/heads/main/docs/assets/instances_val2017.json"
        # zip_path = "annotations_trainval2017.zip"
        extract_path = "./assets"
        json_path = os.path.join(extract_path, "instances_val2017.json")

        # check if json file already exists
        if os.path.exists(json_path):
            return json_path

        # Download the zip file
        print("Downloading annotations ...")
        response = requests.get(url, stream=True)
        with open(json_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Download complete.")

        return json_path

    def __len__(self):
        return len(self.image_urls)

    def __getitem__(self, idx):
        # Get the image url
        return self.getbyindex(idx)

    def getbyindex(self, idx):
        # Get the image url
        url = self.image_urls[idx]
        # Get the image id
        image_id = self.url2imageid[url]
        # Get the bboxes for the image
        bboxes = self.imageid2bbox[image_id]
        # Get the classes for the image
        classes = self.imageid2class[image_id]
        # Download the image
        image = self._download_image(url)
        return image, bboxes, classes

    # write a function to split the dataset into calibration and test sets
    def split(self, test_size=0.8, random_state=42):
        # Split the dataset into calibration and test sets
        calib_urls, test_urls = train_test_split(
            self.image_urls, test_size=test_size, random_state=random_state
        )
        # Create the calibration and test datasets
        calib_dataset = CocoDataset()
        calib_dataset.image_urls = calib_urls
        test_dataset = CocoDataset()
        test_dataset.image_urls = test_urls
        print(f"Calibration dataset size: {len(calib_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        return calib_dataset, test_dataset


class ObjectDetectionAPI:

    def __init__(self, min_iou=0.6):
        """
        Initializes the object detection model and processor with pretrained weights.

        Args:
            min_iou (float, optional): Minimum Intersection over Union (IoU) threshold for matching real and predicted boxes. Defaults to 0.4.

        Attributes:
            model (DetrForObjectDetection): The DETR model initialized with pretrained weights.
            processor (DetrImageProcessor): The image processor for the DETR model initialized with pretrained weights.
            file_path (str): Path to save/load calibration data.
            min_iou (float): Minimum IoU threshold for object detection.
        """
        # Initialize the DETR model and processor from pretrained weights
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", revision="no_timm"
        )
        self.file_path = "calibration_data.pickle"  # Path to save/load calibration data
        self.min_iou = min_iou

    def predict_from_image(self, image):
        # Preprocess the image and make predictions using the model
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])  # Get the size of the image
        bboxes_per_image = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]
        y_preds_per_image = (
            bboxes_per_image["boxes"].detach().numpy()
        )  # Extract bounding boxes
        return y_preds_per_image

    def predict_and_match(self, image, y_trues_per_image):
        # Predict bounding boxes and match them with true bounding boxes using the Hungarian algorithm
        y_preds_per_image = self.predict_from_image(image)
        y_preds_i, y_trues_i, indices_i = hungarian_assignment(
            np.array(y_preds_per_image),
            np.array(y_trues_per_image),
            min_iou=self.min_iou,
        )
        return y_preds_i, y_trues_i, indices_i

    def load_results(self):
        # Load previously saved results from a file
        with open(self.file_path, "rb") as file:
            results_dict = pickle.load(file)
            return (
                results_dict["y_preds"],
                results_dict["y_trues"],
                results_dict["images"],
                results_dict["labels"],
            )

    def save_results(self, y_preds, y_trues, images, labels):
        # Save results to a file
        with open(self.file_path, "wb") as file:
            pickle.dump(
                {
                    "y_preds": y_preds,
                    "y_trues": y_trues,
                    "images": images,
                    "labels": labels,
                },
                file,
            )

    def query(self, dataset, n_instances=None):
        # Initialize lists to store predictions, true labels, images, and classes
        y_preds, y_trues, images, classes = [], [], [], []

        # Check if the file at self.file_path exists
        if os.path.exists(self.file_path):
            y_preds, y_trues, images, labels = self.load_results()
            return y_preds, y_trues, images, labels

        # Iterate over the dataset and predict on each image
        for counter, (image, y_trues_per_image, labels) in enumerate(
            tqdm(dataset), start=1
        ):
            try:
                y_preds_i, y_trues_i, indices_i = self.predict_and_match(
                    image, y_trues_per_image
                )
                y_preds.append(y_preds_i)
                y_trues.append(y_trues_i)
                classes.append(list(compress(labels, indices_i)))
                images.append(image)

                # Break the loop if n_instances is provided
                if n_instances is not None and counter == n_instances:
                    break
            except:
                if n_instances is not None:
                    n_instances = n_instances + 1
                continue

        # Concatenate the lists into arrays
        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)
        classes = np.concatenate(classes)

        # Save the results
        # self.save_results()

        return y_preds, y_trues, images, classes
