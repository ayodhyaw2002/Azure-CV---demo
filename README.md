# Custom Vision Garbage Classifier

This repository demonstrates an **image classification project using Azure Custom Vision** to classify garbage into three categories:

- **Non-Recyclable**  
- **Recyclable**  
- **Organic**  

It includes **test images** and sample images to show how to build, train, and test a model using Azure Custom Vision.

---

## Table of Contents

- [Custom Vision Garbage Classifier](#custom-vision-garbage-classifier)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Create Custom Vision Resources](#create-custom-vision-resources)
  - [Create a Custom Vision Project](#create-a-custom-vision-project)
  - [Upload and Tag Images](#upload-and-tag-images)
  - [Train a Model](#train-a-model)
  - [Test the Model](#test-the-model)
  - [Project Settings](#project-settings)
  - [Running the Demo](#running-the-demo)
    - [Notes](#notes)

---

## Prerequisites

- Python 3.8+ installed  
- `azure-cognitiveservices-vision-customvision` and `msrest` libraries  
- `.env` file containing:  

```text
PredictionEndpoint=<your-prediction-endpoint>
PredictionKey=<your-prediction-key>
ProjectID=<your-project-id>
PublishedIterationName=<your-published-iteration-name>
```

* Test images placed in `test-images/` folder

---

## Create Custom Vision Resources

Before training a model, you need **Azure resources for training and prediction**:

1. Open the [Azure portal](https://portal.azure.com) and sign in.
2. Select **Create a resource**, search for **Custom Vision**, and select it.
3. Create the resource with the following settings:

   * Create options: Both
   * Subscription: Your Azure subscription
   * Resource group: Create or select
   * Region: Any available region
   * Name: A valid name
   * Training pricing tier: F0
   * Prediction pricing tier: F0
4. Wait for deployment to complete. Two resources will be provisioned:

   * Training resource
   * Prediction resource (suffix `-Prediction`)
5. Each resource has its own **endpoint and keys**, which are required for code access.

---

## Create a Custom Vision Project

1. Open the [Custom Vision portal](https://customvision.ai) in a browser.
2. Create a **new project** with these settings:

   * Name: Garbage Classifier
   * Description: Classify garbage images
   * Resource: Your training Custom Vision resource
   * Project Type: Classification
   * Classification Type: Multiclass (single tag per image)
   * Domain: General (or Food/Environment if relevant)

---

## Upload and Tag Images

1. Download or prepare training images in this structure:

```
training-images/
├─ non-recyclable/
├─ recyclable/
├─ organic/
```

2. In the Custom Vision portal, click **Add images**:

   * Upload images from `non-recyclable/` → Tag: `Non-Recyclable`
   * Upload images from `recyclable/` → Tag: `Recyclable`
   * Upload images from `organic/` → Tag: `Organic`

---

## Train a Model

1. Click **Train** in your Custom Vision project.
2. Select **Quick Training** and wait for completion.
3. Review **Precision, Recall, and AP metrics**.

   * Metrics are based on a **50% probability threshold**.
   * You can adjust the threshold at the top-left of the portal.

---

## Test the Model

1. Click **Quick Test** in your project.
2. Upload a test image from `test-images/` or use an image URL.
3. View predictions returned by the model.

   * The class with the highest probability is the predicted label.

---

## Project Settings

* Each project has a **unique Project ID** required for API calls.
* Under **Resources**, note the **endpoint** and **keys** for the training resource.
* Use the **Prediction resource** for inference in your code.

---

## Running the Demo

Example Python snippet:

```python
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os

from dotenv import load_dotenv
load_dotenv()

# Config
prediction_endpoint = os.getenv('PredictionEndpoint')
prediction_key = os.getenv('PredictionKey')
project_id = os.getenv('ProjectID')
published_name = os.getenv('PublishedIterationName')

# Authenticate
credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

# Classify images
for image_file in os.listdir('test-images'):
    with open(os.path.join('test-images', image_file), "rb") as f:
        image_data = f.read()
    results = prediction_client.classify_image(project_id, published_name, image_data)
    for prediction in results.predictions:
        if prediction.probability > 0.5:
            print(image_file, ': {} ({:.0%})'.format(prediction.tag_name, prediction.probability))
```

---

### Notes

* Ensure your iteration is **published** in Custom Vision.
* Training uses **your uploaded images** to fine-tune a pre-trained CNN.
* Prediction uses the **trained iteration** to classify new images.
* Accuracy depends on the **quality, quantity, and tagging** of your images.
