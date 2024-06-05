# Tree Segmentation Project

This repository contains code and resources for a tree segmentation project using U-Net and VGG16. The project focuses on segmenting tree masks in images and includes various datasets and models.

## Project Structure

trees_segmentation_project
├───images
├───kmeans
├───models
└───semantic-drone-dataset
├───dataset
│ └───semantic_drone_dataset
│ ├───label_images_semantic
│ └───original_images
└───RGB_color_image_masks
└───RGB_color_image_masks


- **images**: Directory containing images used for testing.
- **kmeans**: Directory for K-means clustering related files.
- **models**: Directory where trained models are stored.
- **semantic-drone-dataset**: Directory containing the Semantic Drone Dataset.
  - **dataset/semantic_drone_dataset**: Contains label images and original images.
    - **label_images_semantic**: Directory with semantic labels.
    - **original_images**: Directory with original images.
  - **RGB_color_image_masks/RGB_color_image_masks**: Directory with RGB color image masks.

## Dataset

The `semantic-drone-dataset` directory is the main dataset for this project. You must download the Semantic Drone Dataset and place it in the `semantic-drone-dataset` directory as shown in the project structure above. This dataset is essential for training the model.

## Dataset link

http://dronedataset.icg.tugraz.at/

You can use kaggle to donwload it

https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

## Python Version

This project uses Python 3.11.5. Ensure you have this version installed to avoid any compatibility issues.

## Notebooks

The project includes two main Jupyter notebooks:
- `data_prep.ipynb`: Notebook for data preparation.
- `Segmentation.ipynb`: Notebook for training and evaluating the segmentation model.

## Streamlit App

A Streamlit app (`app1.py`) is included to demonstrate how to run the model and visualize the results.

## Usage

1. Clone the repository:

  ```bash
  git clone https://github.com/yourusername/trees_segmentation_project.git
  cd trees_segmentation_project
  ```
2. Install the required dependencies:

It's recommended to use a virtual environment. To create and activate a virtual environment, use the following commands:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

  ```bash
  pip install -r requirements.txt
  ```
3. Prepare the dataset by running the data_prep.ipynb notebook.

4. Train the model by running the Segmentation.ipynb notebook.

5. Run the Streamlit app to visualize the results.(see how to install it below)

### Running the Streamlit App

To run the Streamlit app, follow these steps:

1. Ensure you have Streamlit installed. If not, install it using pip:
  ```bash
   pip install streamlit
  ```
  
2. Run the Streamlit app:
  ```bash
  streamlit run app1.py
  ```

3. Open your web browser and navigate to http://localhost:8501 to view the app.
