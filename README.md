# BrainTumor-Detection

BrainTumor-Detection is an AI model designed to detect brain tumors using image data. This project is developed using TensorFlow, Keras, and Jupyter Python, and it leverages a dataset from Kaggle. The project was created for a hackathon group project.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Project Overview
This project aims to detect brain tumors from MRI images using a deep learning model. The model is built and trained in Google Colab, utilizing TensorFlow and Keras libraries. The dataset used for this project is sourced from Kaggle.

## Installation
Follow these steps to set up the project:

1. **Import the Jupyter Notebook to Google Colab**:
   - Upload the `BrainSpotFinal1.ipynb` notebook to Google Colab.

2. **Mount Google Drive**:
   - Mount your Google Drive to access the dataset and save the model.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Create a Folder on Google Drive**:
   - Create a folder in your Google Drive to store the dataset.

4. **Upload the Dataset**:
   - Upload the dataset from Kaggle to the created folder in your Google Drive.

## Usage
1. **Run the Notebook**:
   - Execute the cells in the notebook from top to bottom. This will involve loading the dataset, preprocessing the data, building the model, training it, and evaluating its performance.

2. **Validation and Accuracy Graphs**:
   - After training, the notebook will generate validation and accuracy graphs to help visualize the model's performance.

3. **Testing the Model**:
   - Get the image link from the test folder.
   - Set the path for the test image and run the respective cell to see the prediction preview.

    ```python
    test_image_path = '/content/drive/MyDrive/your_test_folder/test_image.jpg'
    ```

4. **Run the Test Cell**:
   - Ensure you have set the correct path for your test image and execute the cell to get the prediction.

## Contributing
We welcome contributions to enhance the functionality and performance of this project. Here are some ways you can contribute:

- Report bugs and issues
- Suggest new features or enhancements
- Submit pull requests for bug fixes and new features

## Acknowledgements
- **Team Members**:
  - Pasan Pitigala
  - Akila Kasun
  - Janindu Himansa
  - Nethni Dias
  - Gayan Sachintha

- **Dataset**:
  - The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/).

- **Libraries**:
  - TensorFlow
  - Keras
  - Jupyter Notebook

Thank you for your interest in the BrainTumor-Detection project. We hope this tool proves to be valuable in detecting brain tumors and advancing medical research.
