# Balancing Real and Synthetic Data for Age Estimation and Verification
This repository comprises our exploration of synthetic image generation to help balance and improve age estimation techniques. 
## Predictor
Python scripts for building datasets, training, and testing regression models can be found in the predictor subdirectory.
* csv_builder.py - Generate a .csv file representative of a dataset contained within a series of subdirectories. Subdirectories must be labelled with the age contained within given folder (./35 : directory containing images of 35 year olds)
* model.py - The trained model used. Using EfficientNet_B0 to build a CNN age estimation regressor. Requires ages normalized to 0-1.
* train.py - Train regression model on given .csv file and save the result after 10 epochs. Used to train a regression model from a given .csv generated with csv_builder.py
* test.py - Evaluate models generated with train.py. Evaluted on both regression metrics and classification metrics with a given threshold. Metrics are saved and outputted to two files containing summary metrics and per age classification metrics, for further inspection and analysis.
## Trained Models
Included are pretrained regression models with various levels of synthetic augmentation: 

* uniform100.pth - 100% real dataset with filtering and balancing
* styleganUniform90.pth - 10% synthetically augmented dataset using StyleGan3
* samUniform90.pth - 10% synthetically augmented dataset using SAM
* styleganUniform80.pth - 20% synthetically augmented dataset using StyleGan3
* samUniform80.pth - 20% synthetically augmented dataset using SAM 
* styleganUniform70.pth - 30% synthetically augmented dataset using StyleGan3

## How to Use
1. Build a dataset using balance_app.py, specifying synthetic and real image directories
2. Generate a .csv from dataset using csv_builder.py
3. Use train.py to train a specified model, using the .csv file generated from csv_builder.py
4. Use test.py to evaluate model results after training and save measured metrics.