import torch
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw
import pytesseract
import requests
import os 
from llm import inference, upload_image

import re


cropped_images_dir = "cropped_images"
os.makedirs(cropped_images_dir, exist_ok=True)

# Load YOLO model
class YOLOModel:
    def __init__(self, model_path="yolov5s.pt"):
        """
        Initialize the YOLO model. Downloads YOLOv5 pretrained model if not available.
        """
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
        # self.model2 = YOLOv10.from_pretrained("Ultralytics/Yolov8")
        # print(f'YOLO Model:\n\n{self.model}')
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # # print(f'CLIP Model:\n\n{self.clip_model}')
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.category_brands = {
        #     "electronics": ["Samsung", "Apple", "Sony", "LG", "Panasonic"],
        #     "furniture": ["Ikea", "Ashley", "La-Z-Boy", "Wayfair", "West Elm"],
        #     "appliances": ["Whirlpool", "GE", "Samsung", "LG", "Bosch"],
        #     "vehicles": ["Tesla", "Toyota", "Ford", "Honda", "Chevrolet"],
        #     "chair": ["Ikea", "Ashley", "Wayfair", "La-Z-Boy", "Herman Miller"],
        #     "microwave": ["Samsung", "Panasonic", "Sharp", "LG", "Whirlpool"],
        #     "table": ["Ikea", "Wayfair", "Ashley", "CB2", "West Elm"],
        #     "oven": ["Whirlpool", "GE", "Samsung", "Bosch", "LG"],
        #     "potted plant": ["The Sill", "PlantVine", "Lowe's", "Home Depot", "UrbanStems"],
        #     "couch": ["Ikea", "Ashley", "Wayfair", "La-Z-Boy", "CushionCo"],
        #     "cow": ["Angus", "Hereford", "Jersey", "Holstein", "Charolais"],
        #     "bed": ["Tempur-Pedic", "Ikea", "Sealy", "Serta", "Sleep Number"],
        #     "tv": ["Samsung", "LG", "Sony", "Vizio", "TCL"],
        #     "bin": ["Rubbermaid", "Sterilite", "Hefty", "Glad", "Simplehuman"],
        #     "refrigerator": ["Whirlpool", "GE", "Samsung", "LG", "Bosch"],
        #     "laptop": ["Dell", "HP", "Apple", "Lenovo", "Asus"],
        #     "smartphone": ["Apple", "Samsung", "Google", "OnePlus", "Huawei"],
        #     "camera": ["Canon", "Nikon", "Sony", "Fujifilm", "Panasonic"],
        #     "toaster": ["Breville", "Cuisinart", "Black+Decker", "Hamilton Beach", "Oster"],
        #     "fan": ["Dyson", "Honeywell", "Lasko", "Vornado", "Bionaire"],
        #     "vacuum cleaner": ["Dyson", "Shark", "Roomba", "Hoover", "Bissell"]
        # }


    def predict_clip(self, image, brand_names):
        """
        Predict the most probable brand using CLIP.
        """
        inputs = self.clip_processor(
            text=brand_names,
            images=image,
            return_tensors="pt",
            padding=True
        )
        # print(f'Inputs to clip processor:{inputs}')
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities
        best_idx = probs.argmax().item()
        return brand_names[best_idx], probs[0, best_idx].item()


    def predict_text(self, image):
        grayscale = image.convert('L')
        text = pytesseract.image_to_string(grayscale)
        return text.strip()


    def predict(self, image_path):
        """
        Run YOLO inference on an image.

        :param image_path: Path to the input image
        :return: List of predictions with labels and bounding boxes
        """
        results = self.model(image_path)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        predictions = results.pandas().xyxy[0]  # Get predictions as pandas DataFrame
        print(f'YOLO predictions:\n\n{predictions}')
        output = []
        for idx, row in predictions.iterrows():
            category = row['name']
            confidence = row['confidence']
            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]

            # Crop the detected region
            cropped_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            cropped_image_path = os.path.join(cropped_images_dir, f"crop_{idx}.jpg")
            cropped_image.save(cropped_image_path, "JPEG")

            # uploading to cloud for getting URL to pass into LLM
            print(f'Uploading now to image url')
            image_url = upload_image.upload_image_to_imgbb(cropped_image_path)
            print(f'Image URL received as{image_url}')
            # inferencing llm for possible brands
            result_llms = inference.get_name(image_url, category)
            # possible_brands_llm = re.findall(r"-\s*(.+)", possible_brands_mixed)

            # if len(possible_brands_llm)>0:
            #     predicted_brand, clip_confidence = self.predict_clip(cropped_image, possible_brands_llm)
            # else:
            #     predicted_brand, clip_confidence = "Unknown", 0.0
            

            '''
            # Match category to possible brands
            if category in self.category_brands:
                possible_brands = self.category_brands[category]
                print(f'Predicting with CLIP:\n\n')
                predicted_brand, clip_confidence = self.predict_clip(cropped_image, possible_brands)
            else:
                predicted_brand, clip_confidence = "Unknown", 0.0
            '''


            detected_text = self.predict_text(cropped_image)
            print(f'Details:{detected_text}')
            print(f'Predicted brand: {result_llms["model"]}')
            # Draw bounding box and label on the image
            draw.rectangle(bbox, outline="red", width=3)
            draw.text(
                (bbox[0], bbox[1] - 10),
                f'{result_llms["brand"]})',
                fill="red"
            )

            # Append result
            output.append({
                "category": category,
                "bbox": bbox,
                "confidence": confidence,
                "category_llm":result_llms["brand"],
                "predicted_brand": result_llms["model"],
                # "clip_confidence": clip_confidence,
                "price":result_llms["price"],
                "details":result_llms["description"],
                "detected_text":detected_text,
            })

            valid_indices = set(range(len(predictions)))

            # Iterate over all files in the directory
            for filename in os.listdir(cropped_images_dir):
                # Check if the filename matches the pattern for cropped images
                if filename.startswith("crop_") and filename.endswith(".jpg"):
                    # Extract the index from the filename
                    try:
                        file_idx = int(filename.split("_")[1].split(".")[0])
                        if file_idx not in valid_indices:
                            # Delete the file if its index is not valid
                            file_path = os.path.join(cropped_images_dir, filename)
                            os.remove(file_path)
                            print(f"Deleted excess file: {filename}")
                    except ValueError:
                        # Skip files that don't match the pattern
                        continue

        return output


        