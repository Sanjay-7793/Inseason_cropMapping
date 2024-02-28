import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal as gd
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#
# Load data set
#
inpCSV = r"F:\Research_work\GT\GT_01_2024\Feb_sfn_2022\Feb_2FN_trainingdata.csv"    # input labelled training data
data =pd.read_csv(inpCSV)

#Reading the band values
X = data.iloc[:, 1:12]

#Reading the class 
y = data.iloc[:,-1:]

# Seperating the testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

y_train = y_train.values.ravel()
#X_train = X_train.values.ravel()

# Create an instance of Random Forest Classifier
forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=100,
                                 random_state=1,
                                 n_jobs=2)

#
# Fit the model
#
y
forest.fit(X_train, y_train)

#
# Measure model performance
#
y_pred = forest.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

##Open the raster data 

inpRaster = r"F:\Rabi_2023_24\Bands_NDVI_LS\Feb2FN_preprocessedimage.tif"   #add the location of pre-processed input data
ds = gd.Open(inpRaster,gd.GA_ReadOnly)

#get raster information from the above image
cols = ds.RasterXSize
rows = ds.RasterYSize
bands = ds.RasterCount
geo_transform = ds.GetGeoTransform()
projection = ds.GetProjectionRef()

##Converting Raster Data as Array
array = ds.ReadAsArray()
array = np.nan_to_num(array)
print(array)
ds = None
##Reshaping the array and converting to dataframe
array = np.stack(array,axis=2)
array = np.reshape(array, [rows*cols,bands])
test = pd.DataFrame(array, dtype='float32')

del array

#Classification of dataset using Random Forest
class_image = forest.predict(test.values)

class_image.shape

## Removal of the classes from the non-image pixels in the extend
sum_row = test.sum(axis=1)

for i, v in sum_row.items():
 if v == 0:
   class_image[i] = 0

## Reshaping the Output
output = class_image.reshape((rows,cols))
output.shape

#defining location
outRaster = r"F:\Rabi_2023_24\Classified\Feb_SFN\Feb2FN_classified.tif"     #set the location and name of classified image
##Create the geotiff
def createGeotiff(outRaster, data, geo_transform, projection):
    # Create a GeoTIFF file with the given data
    driver = gd.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gd.GDT_Int32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    rasterDS = None


#export classified image
createGeotiff(outRaster,output,geo_transform,projection)