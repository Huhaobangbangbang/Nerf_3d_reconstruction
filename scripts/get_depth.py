from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests


def get_depth(image_path):

    image = Image.open(image_path)
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    # formatted = (output * 255 / np.max(output)).astype("uint8")
    # depth = Image.fromarray(formatted)
    return output




def get_depth2():
    from transformers import pipeline
    estimator = pipeline("depth-estimation")
    result = estimator('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1.jpg')
    
depth = get_depth('/cloud/private/huh/scripts/biyelunwen/scale_estimation/dataset/1.jpg')
print(depth.shape())


