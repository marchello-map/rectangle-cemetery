import cv2 as cv
import numpy as np
import random as rng
from osgeo import gdal
import os
import ogr

rng.seed(12345)

def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = []
    min_rect = [None] * len(contours)
    for i, c in enumerate(contours):
        epsilon = 0.025*cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        if cv.arcLength(c, True) > 500:
            contours_poly.append(approx)
            min_rect[i] = cv.minAreaRect(c)

    #drawing
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # Draw polygonal contour + bounding rects + circles
    for i in range(len(contours)):
        cv.drawContours(drawing, contours_poly, -1, (0,255,0), 1)
        # rotated rectangle
        box = cv.boxPoints(min_rect[i])
        box = np.intp(box)
        cv.drawContours(drawing, [box], 0, (0,0,255), 2)

    cv.imshow('Contours', drawing)
    #writes a plain tiff
    outlines = "sirok_box_1.tif"
    cv.imwrite(outlines, drawing)

    # Read band 1 (R) of outlines and write out as gtiff
    outlines_a = gdal.Open(outlines)
    band = outlines_a.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    arr_mean = int(arr.mean())
    arr_out = np.where((arr < arr_mean), 1000, arr)
    driver_gt = gdal.GetDriverByName("GTiff")
    outdata = driver_gt.Create(outFileName, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(src_a.GetGeoTransform())
    outdata.SetProjection(src_a.GetProjection())
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.FlushCache()

    #Write band 1 to shp
    outShapefile = "polygonized"
    driver_shp = ogr.GetDriverByName("GeoJSON")
    if os.path.exists(outShapefile+".geojson"):
        driver_shp.DeleteDataSource(outShapefile+".geojson")
    outDataSource = driver_shp.CreateDataSource(outShapefile+".geojson")
    outLayer = outDataSource.CreateLayer("polygonized", srs = None)
    gdal.Polygonize(band, None, outLayer, -1, [], callback=None)

#Read source img with cv2
#src = cv.imread('sirok_gt.tif')
# Read source img with gdal and pass it to opencv
file = "sirok_gt.tif"
(fileRoot, fileExt) = os.path.splitext(file)
outFileName = fileRoot + "_mod" + fileExt
src_a = gdal.Open(file)
src_b = src_a.ReadAsArray()
src = np.dstack((src_b[0],src_b[1],src_b[2]))
src = cv.cvtColor(src, cv.COLOR_RGB2BGR)

# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))

# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)

#trackbar in the window
max_thresh = 255
thresh = 80 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)

cv.waitKey()