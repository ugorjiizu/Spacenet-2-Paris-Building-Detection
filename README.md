# Spacenet-2-Paris-Building-Detection
![image](https://github.com/ugorjiizu/Spacenet-2-Paris-Building-Detection/assets/66518563/fb8df2a6-4f0d-4a59-bc6c-2204abe1b004)

## Project Overview

The commercialization of the geospatial industry has led to an explosive amount of data being collected to characterize our changing planet. One area for innovation is the application of computer vision and deep learning to extract information from satellite imagery at scale. CosmiQ Works, Radiant Solutions and NVIDIA have partnered to release the SpaceNet data set to the public to enable developers and data scientists to work with this data.

The SpaceNet 2 Building Detection Project is designed to accurately identify and mask buildings in satellite images using advanced deep learning techniques. Leveraging the powerful ResNet segmentation model, this project processes satellite imagery from the SpaceNet 2 dataset to create precise building masks. These masks can be used for various applications, including urban planning, disaster response, and geographic information systems (GIS).

## Data

### SpaceNet 2 Dataset

The SpaceNet 2 dataset consists of high-resolution satellite images covering various urban areas. It includes annotations for building footprints, which are used for training and evaluation in this project.

#### Sample Images and Masks

Below are sample images from the SpaceNet 2 dataset along with their corresponding masks:

![image](https://github.com/ugorjiizu/Spacenet-2-Paris-Building-Detection/assets/66518563/c75c5fee-69de-4c2a-8b1a-c25c1eac67b2)

These samples demonstrate the input satellite images and their respective ground truth masks used for training the building detection model.

## Key Components

- **Data Processing**: Preprocessing and augmentation of satellite images from the SpaceNet 2 dataset.
- **Model Architecture**: Implementation of a ResNet-based segmentation model to detect building footprints.
- **Training and Evaluation**: Techniques to train the model, evaluate its performance, and refine its accuracy.
- **Visualization**: Tools to visualize the masked images and assess the quality of the building detection.
  
This project showcases the effectiveness of convolutional neural networks (CNNs) in the domain of satellite imagery analysis and provides a robust framework for future research and development in building detection.

You can either run this on the kaggle environment, where you just need to create a copy and run both notebooks https://www.kaggle.com/code/ugorjiir/build-detect-paris or follow the instructions below

## Installation Instructions

To create and run the building detection model, we provide two Jupyter notebooks for your convenience:

1. **Training Notebook**: This notebook is designed for training the ResNet segmentation model using the SpaceNet 2 dataset.
2. **Inference Notebook**: This notebook allows you to perform inference with the trained model, generating building masks from new satellite images.

### Steps to Set Up

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/spacenet2-building-detection.git
   cd spacenet2-building-detection
   
2. **Install Dependencies**:
   Ensure you have Python and pip installed. Then, install the required libraries:
   pip install -r bd_requirements.txt
   
3. **Download the SpaceNet 2 Dataset**:
   The dataset is available in the Kaggle environment. If you require a larger dataset beyond the Paris Area of Interest (AOI) used for training the model, visit the [SpaceNet 2 Dataset](https://spacenet.ai/spacenet-buildings-dataset-v2/) page to download additional data from SpaceNet.

4. **Train the Model**:
   Open the `build_detect_paris.ipynb` and follow the instructions to train the model on the SpaceNet 2 dataset.
   
5. **Inference**:
   Once the model is trained, open the model inference notebook to generate building masks from the testing data for satellite images.
   Also, please note you might need to download the outputs from the training section and specify your model path for inference


## Model Performance
The performance of the ResNet segmentation model is evaluated using standard metrics such as Intersection over Union (IoU). The model achieves an IoU of 63% on the test dataset. These metrics indicate the model's ability to accurately detect and segment building footprints in satellite images.
![image](https://github.com/ugorjiizu/Spacenet-2-Paris-Building-Detection/assets/66518563/3d1be0dd-25b7-4662-94bc-ca90385601a7)

Features
High accuracy in building detection using ResNet segmentation.
Flexible data processing pipeline for satellite imagery.
Comprehensive training and evaluation framework.
Visualization tools for assessing model performance and results.

### Technologies Used
- Python
- Pytorch
- OpenCV
- Jupyter Notebook
- Pandas & Geopandas

