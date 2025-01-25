import gdown
from pathlib import Path
import os
import zipfile
from dotenv import load_dotenv
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import shutil

class LubinaDatasetProcessor:
    def __init__(self, file_id: str = os.getenv("LUBINA_GOOGLE_DRIVE_FILE_ID")):
        self.file_id = file_id
        self.dataset_path = Path(os.path.join(os.getcwd(), "datasets/Lubina"))
        self.merged_dataset_path = Path(os.path.join(os.getcwd(), "datasets/Marged/train"))
        
    def download_and_extract(self):
        url = f'https://drive.google.com/uc?id={self.file_id}'
        
        if (not self.dataset_path.exists()):
            self.dataset_path.mkdir()

        output = self.dataset_path / 'PO.v2i.coco-segmentation.zip'
        gdown.download(url, str(output), quiet=False)

        with zipfile.ZipFile(output, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if not file.endswith('.txt'):
                    zip_ref.extract(file, self.dataset_path)

        os.remove(output)

    def handle_robflow_dataset(self, paths: list[Path]):
        for path in paths:
            coco_path = path / Path('_annotations.coco.json')
            coco = COCO(coco_path)
            img_dir = path
            image_id = 0

            for image_id in coco.imgs:
                img = coco.imgs[image_id]
                image_path = os.path.join(img_dir, img['file_name'])
                original_image = Image.open(image_path)
                image = np.array(original_image)

                # Utwórz czarny obraz o tych samych wymiarach co oryginalny obraz
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

                # Pobierz ID kategorii i anotacji
                cat_ids = coco.getCatIds()
                anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
                anns = coco.loadAnns(anns_ids)

                # Narysuj anotacje na czarnym obrazie
                for ann in anns:
                    mask = np.maximum(mask, coco.annToMask(ann) * 255)

                mask_image = Image.fromarray(mask)
                labels_dir = path / 'labels'
                labels_dir.mkdir(parents=True, exist_ok=True)
                
                # Użyj unikalnej nazwy opartej na nazwie pliku obrazu
                mask_filename = f"{img['file_name'].split('.')[0]}_mask.png"
                mask_image.save(os.path.join(labels_dir, mask_filename))
                
                images_dir = path / 'images'
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # Użyj tej samej nazwy dla oryginalnego obrazu
                original_image.save(os.path.join(images_dir, f"{img['file_name'].split('.')[0]}.png"))
                os.remove(image_path)

            os.remove(coco_path)

    def merge_datasets(self):
        images_merged_dir = self.merged_dataset_path / 'images'
        labels_merged_dir = self.merged_dataset_path / 'labels'

        if not images_merged_dir.exists():
            images_merged_dir.mkdir(parents=True, exist_ok=True)
        if not labels_merged_dir.exists():
            labels_merged_dir.mkdir(parents=True, exist_ok=True)

        for folder in ['train', 'test', 'valid']:
            images_dir = self.dataset_path / folder / 'images'
            labels_dir = self.dataset_path / folder / 'labels'

            for image_file in images_dir.glob("*.png"):
                shutil.move(str(image_file), images_merged_dir / image_file.name)

            for label_file in labels_dir.glob("*.png"):
                shutil.move(str(label_file), labels_merged_dir / label_file.name)

    def process(self):
        self.download_and_extract()

        # Ścieżki do folderów z danymi
        dataset_folders = ['test', 'train', 'valid']
        for folder in dataset_folders:
            dataset_path = self.dataset_path / folder
            self.handle_robflow_dataset([dataset_path])

        self.merge_datasets()
        # shutil.rmtree(self.dataset_path)