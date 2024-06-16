import gradio as gr
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
import numpy as np
from PIL import Image

# Load the model from the local path
model_path = "/home/dev/synth/projects/beagle_security-intern_test/src/alz_model"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

# Get the label names from the model's configuration
labels = model.config.id2label

print("Labels:", labels)  # Debugging statement to check the labels

# Define the prediction function (with preprocessing)
def predict_image(image):
    """
    Predicts the Alzheimer's disease stage from an uploaded MRI image.

    Args:
        image: The uploaded MRI image (PIL Image).

    Returns:
        The predicted label with its corresponding probability.
    """
    # Preprocessing steps:
    image = np.array(image)
    if image.ndim == 2:  # Convert grayscale to RGB if needed
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # Resize image if necessary (optional step)
    if image.shape[0] != 224 or image.shape[1] != 224:
        image = np.array(Image.fromarray(image).resize((224, 224)))

    # Model inference:
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    print(f"logits shape: {logits.shape}")  # Debugging statement to check shape
    print(f"logits: {logits}")  # Debugging statement to check content

    predicted_label_id = logits.argmax(-1).item()
    predicted_label = labels[predicted_label_id]

    # Calculate probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidences = {labels[i]: float(probabilities[0][i]) for i in range(len(labels))}

    return predicted_label, confidences

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.JSON(label="Confidence Scores")
    ],
    title="Alzheimer's Disease MRI Image Classifier",
    description="Upload an MRI image to predict the stage of Alzheimer's disease."
)

iface.launch()
