 
<font color="green">  ## GAN-HTR </font>
### Description
This is an implementation for the paper "Enhance to Read Better: A Multi-Task Adversarial Network for Handwritten Document Image Enhancement" designed to enhance the document quality before the recognition process. It could be used for document cleaning and binarization. 


### License
This work is only allowed for academic research use. For commercial use, please contact the author.

### Requirements

install the requirements.txt
pip install  requirements.txt

### Insert Artficial distortion on images

python distort_image_khatt.py

###  Train a HTR system for a separate task

python train_khatt_basic_distorted.py

###  Train the GAN-HTR using an text line image database

python GAN_AHTR.py

###  Document binarization

python eval_Dibco_2010.py

###  Image

![H03](https://user-images.githubusercontent.com/15616524/148749752-88e0661f-4356-45f5-b1b1-bc34cd872164.png)

###  Binarzed Image

![b_predicted_3](https://user-images.githubusercontent.com/15616524/148748926-a264adbd-ea5b-4470-b9a2-349318368a80.png)


###  Citation:

If this work was useful for you, please cite it as:

@article{KHAMEKHEMJEMNI2022108370,
title = {Enhance to read better: A Multi-Task Adversarial Network for Handwritten Document Image Enhancement},
journal = {Pattern Recognition},
volume = {123},
pages = {108370},
year = {2022},
doi = {https://doi.org/10.1016/j.patcog.2021.108370},
author = {Sana {Khamekhem Jemni} and Mohamed Ali Souibgui and Yousri Kessentini and Alicia Fornés},
}
  
  
