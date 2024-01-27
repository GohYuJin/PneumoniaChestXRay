import torch
import requests
from PIL import Image
import torchvision
from transformers import AutoModelForImageClassification
import gradio as gr

model = AutoModelForImageClassification.from_pretrained("TriEightz/PneumoniaChestXRay-ConvNextBase", trust_remote_code=True)

labels = ["Normal", "Pneumonia"]

image_size = 384
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(image_size, image_size)),
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.ToTensor(),
])

def predict(inp):
    inp = transforms(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)['logits'][0, 0:2], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(2)}
    return confidences

title = "Pneumonia Chest X-Ray Classifier"

desc = "This is a ConvNext trained for the Pneumonia Chest X-Ray dataset. It can help to determine whether the patient has pneumonia or not through passing in a Chest X-Ray Image .The example images provided are all taken from the unseen test set. The first 3 images are from patients without pneumonia, while the last 3 images were from patients diagnosed with pneumonia"

long_desc = "This is a ConvNext trained for the Pneumonia Chest X-Ray dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) achieving 97% accuracy on the test set. The dataset contains a total of 5,856 chest X-ray images from children patients in the Guangzhou Women and Children’s Medical Center. We retain the original training and testing split that the authors used. Thus, 5,232 of the images were used for training and 624 images were used for testing. 3,883 of the training images contain examples with pneumonia present and the remaining 1,349 training chest X-ray images have been determined to be free of Pneumonia"

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3),
             examples=["Normal (1).jpeg", "Normal (2).jpeg", "Normal (3).jpeg", "Pneumonia (1).jpeg", "Pneumonia (2).jpeg", "Pneumonia (3).jpeg"],
             title=title, 
             description=desc, 
             article=long_desc).launch()