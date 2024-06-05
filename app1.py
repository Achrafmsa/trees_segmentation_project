import streamlit as st
import torch
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Function to load model
def load_model(uploaded_file):
    with open('temp_model.pt', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    model = torch.load('temp_model.pt')
    return model

# Function to predict mask
def predict_mask(model, image, transform, device):
    model.eval()  # Set the model to evaluation mode

    image = Image.open(image).convert("RGB")
    image = np.array(image)
    
    # Apply the same transformations as during training
    augmented = transform(image=image)
    image = augmented['image']
    image = T.ToTensor()(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()  # Binarize the output
        output = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU

    return output

# Streamlit interface
st.title("U-Net Model Image Segmentation")

# Upload the model
uploaded_model = st.file_uploader("Upload a trained U-Net model", type=["pt"])

if uploaded_model is not None:
    model = load_model(uploaded_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_prediction = A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Upload an image
    uploaded_image = st.file_uploader("Upload an image to segment", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        
        # Predict the mask
        predicted_mask = predict_mask(model, uploaded_image, transform_prediction,device)

        # Display the input image and the predicted mask
        st.write("## Input Image")
        st.image(Image.open(uploaded_image), caption="Uploaded Image", use_column_width=True)
        
        st.write("## Predicted Mask")
        plt.figure(figsize=(6, 6))
        plt.imshow(predicted_mask, cmap='gray')
        st.pyplot(plt)
