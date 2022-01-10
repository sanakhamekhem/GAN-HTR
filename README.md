# GAN-HTR
This is an implementation for the paper Enhance to Read Better: A Multi-Task Adversarial Network for Handwritten Document Image Enhancement designed to enhance the document quality before the recognition process. It could be used for document cleaning andbinarization. The weights are available to test the enhancement.


# License
This work is only allowed for academic research use. For commercial use, please contact the author.

# Requirements

install the requirements.txt

# Insert Artficial distortion on images

python distort_image_khatt.py

# Train a HTR system for a separate task

python train_khatt_basic_distorted.py

# Train the GAN-HTR using an text line image database

python GAN_AHTR.py

# Document binarization

python eval_Dibco_2010.py

# Image



# Binarzed Image

![b_predicted_3](https://user-images.githubusercontent.com/15616524/148748926-a264adbd-ea5b-4470-b9a2-349318368a80.png)
