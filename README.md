## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
3. [Usage](#usage)

### General Info
***
Gaining insight into engineering material behaviours and properties has become highly dependent on microscopy analysis
in the modern world.
While obtaining high-quality micrographs is a challenge in itself, a common difficulty is processing and analysing the
data thereafer. The purpose of this project is to analyse light-microscopy images using pixel intensity data, explore
the data and use it to train various Machine Learning (ML) models. The microscopy images were taken from Aluminum Alloy AA3104
after different heat-treatments and the aim of the project is to train ML models to accurately categorise the heat-treatment
based only on image(pixel) data.     

## Technologies
***
A list of technologies used within the project:
* [Pandas](https://pandas.pydata.org/): Version 0.25.3 
* [Numpy](https://numpy.org/): Version 1.18.1
* [Matplotlib](https://matplotlib.org/): Version 3.1.2
* [openCV](https://pypi.org/project/opencv-python/): Version 4.4.0
* [openpyxl](https://pypi.org/project/opencv-python/): Version 3.0.5
* [seaborn](https://seaborn.pydata.org/): Version 0.11.1
* [sklearn](https://scikit-learn.org/stable/): Version 0.24.1
## Usage
This projects consists of 3 modules which handle the raw image processing; clustering analyses and Machine Learning anaysis.
In addition, an example script has been included to demonstrate various outcomes of the project.
1. Image processing: 'Image_processing.py'
2. Clustering analysis: 'Image_clustering.py'
3. Machine Learning: 'ML_Image_Calssification.py'
4. Example script: 'Example_Image_processing.py' 