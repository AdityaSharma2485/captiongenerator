# Image Captioning with Transformer - Flickr8k Dataset

![1](https://github.com/AdityaSharma2485/captiongenerator/assets/92670331/2fde79bd-a8d0-415c-8b00-54f02a15a7bc)


## Description

This GitHub repository contains an image captioning model trained on the Flickr8k dataset using the Transformer architecture. The model can generate descriptive captions for images, providing a human-like understanding of the visual content.

## Key Features

- **Transformer-Based Architecture:** The model is built upon the powerful Transformer architecture, which has shown significant success in natural language processing tasks and is adapted for image captioning.

- **CNN Encoder:** The image captioning process starts with a Convolutional Neural Network (CNN) encoder based on the InceptionV3 pre-trained model. The encoder extracts meaningful image features used to create a visual representation of the input image.

- **Transformer Encoder and Decoder:** The Transformer-based encoder and decoder layers process the image features and generate captions. The encoder encodes the image information into a fixed-length representation, while the decoder generates the sequence of words for the caption.

- **Data Augmentation:** During training, the model benefits from data augmentation techniques to enhance generalization. Random flips and brightness adjustments are applied to images for added robustness.

- **Vocabulary and Tokenization:** The captions are tokenized and converted into sequences of indices using TensorFlow's TextVectorization and StringLookup layers. This allows for efficient and consistent text processing.

- **Training and Evaluation:** The model is trained using Sparse Categorical Crossentropy loss and optimized using the Adam optimizer. Early stopping is applied to prevent overfitting. The model's performance is evaluated on a separate validation dataset.

## Dataset

The Flickr8k dataset consists of 8,000 images, each paired with five human-generated captions. This diverse and well-labeled dataset provides ample training data for the image captioning model.

## Usage

1. Clone the repository and set up the required dependencies.
2. Download the Flickr8k dataset and pre-process the captions and images as described in the code. Link to dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
3. Train the image captioning model using the provided Python jupyter notebook.
4. Evaluate the model's performance on the validation dataset.
5. Use the trained model to generate captions for new images.

## Contributions

Contributions to the project, such as bug fixes, enhancements, and new features, are welcome. Please create a pull request with a detailed description of your changes.

## Acknowledgments

Special thanks to the original authors of the Transformer architecture and the Flickr8k dataset creators for their valuable contributions to the field of natural language processing and computer vision.

### Note:

This project was created as part of a personal learning project and may be improved or extended over time. If you find this repository helpful, consider giving it a star to show your appreciation!

