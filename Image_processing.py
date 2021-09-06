""" Loads and processes JPEG images to pixels for further analysis"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from os import listdir

class Images_Preprocessing:
    def __init__(self, directories):
        self.directories = directories
        self.images_df = None

    def imageslist(self):
        imlist = []
        # Create a list to store labels/ discriptions of each image
        imlabellist = []
        # Create a dict to store data for each image
        image_dict = {}
        for i, directory in enumerate(self.directories):
            for file in listdir(directory): # Loop over all files in directory
                # Create variable for image
                im = imread(directory + '/' + str(file))
                # Convert image Grayscale
                imgray = cvtColor(im, COLOR_BGR2GRAY)
                # Group pixels by intensity
                counts, bins = np.histogram(imgray, range(256))
                image_dict[str(i) + '_' + file] = counts
                imlist.append(im)
                imlabellist.append(directory)
            imarray = np.stack(imlist, axis=3)
            imlabelarray = np.array(imlabellist)
        # Create a Dataframe with all the image data and associated labels
        df = pd.DataFrame(image_dict).transpose()
        df.columns = ["P"+str(x) for x in range(0, len(df.columns)) ]
        df['y'] = imlabellist
        self.images_df = df
        q = input('Save image data to csv? Y/N')
        if q=='Y':
            file_name = input('Enter file name:')
            df.to_csv('file_name',index = None)
        return imarray, imlabelarray, imlist, df

if __name__ == "__main__":
    main()