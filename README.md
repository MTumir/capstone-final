# Face Region Segmentation for Privacy-Preserving Age Identification
This repository comprises my exploration of synthetic image generation to help balance and improve age estimation techniques. 
Find the original repository here: [Balancing Real and Synthetic Data for Age Estimation and Verification](https://github.com/kadendavis866/capstone-final)

## Dataset
Scripts for balancing, filtering, and generating data for experiments.

### Filtering
* yunet_filter.py - Uses YuNet to filter images into 'good' and 'bad' directories based upon their confidence score.

### Miscellaneous
* image_dimension_lister.py - Logs the resolution of provided images to a file.

## SHAP
Scripts for determining the most important facial features via Shapley values.
* shap_age_predictor.py - Uses the confidence scores from a model trained from predictor/train_model.py to determine which regions of the face influence classification the most.
* shap_yunet.py - Uses the confidence scores from YuNet to determine which regions of the face influence facial recognition the most.

## Face Region Extractor
Scripts for extracting face regions from input images.
* disk_facial_region_extractor.py - Takes a directory of images as input, and creates a mirrored directory of the same photos, cropped to only include a selected feature (eye, nose, mouth).
* runtime_facial_region_extractor.py - Takes a directory of images as input, and returns the next image when calling next_processed_image(), cropped to only include a selected feature (eye, nose, mouth, forehead, cheek). Also includes process_image(), to extract a feature from a single provided image.

## Predictor
...

## Trained Models
The following trained models are included in predictor/trained_models: 
* ...

## How to Use
1. ...