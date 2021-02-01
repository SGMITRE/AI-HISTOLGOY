import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy.ndimage.morphology import binary_fill_holes
from scipy import linalg
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.color import separate_stains
from skimage.morphology import binary_closing, disk, binary_opening

def optical_density(tile):
    """
    Convert a tile to optical density values.

    Parameters
    ----------
    tile : numpy array
        RGB image.
    """
    tile = tile.astype(np.float64)
    od = -np.log((tile+1)/255 + 1e-8)
    return od

def find_tissue(tile):
    """
    Segments the tissue and calculates the proportion of the image contaning tissue

    Parameters
    ----------
    tile : numpy array
        RGB image.
    """


    # Convert to optical density values
    tile = optical_density(tile)

    # Threshold at beta and create binary image
    beta = 0.12
    tile = np.min(tile, axis=2) >= beta

    # Remove small holes and islands in the image
    tile = binary_opening(tile, disk(3))
    tile = binary_closing(tile, disk(3))

    # Calculate percentage of tile containig tissue
    percentage = np.mean(tile)
    tissue_amount = percentage #>= tissue_threshold

    return tissue_amount, tile

def color_deconvolution(tile):
    """
    Applies color deconvolution to separate the RGB images into grayscale
    images representing the concetration of hematoxylin and eosin.

    Parameters
    ----------
    tile : numpy array
        RGB image.
    """

    ## Define stain color vectors
    rgb_from_hed = np.array([[0.650, 0.704, 0.286],
                            [0.07, 0.99, 0.11],
                            [0.0, 0.0, 0.0]])
    rgb_from_hed[2, :] = np.cross(rgb_from_hed[0, :], rgb_from_hed[1, :])
    hed_from_rgb = linalg.inv(rgb_from_hed)
    ihc_hed = separate_stains(tile, hed_from_rgb)

    ## Normalize images
    for i in np.arange(0,3):
        im = ihc_hed[:, :, i]
        im = im - np.min(im)
        im = im / np.max(im)
        im = np.abs(im-1)*255
        im = np.round(im)
        ihc_hed[:, :, i] = im
    im_stains = ihc_hed.astype(np.uint8)

    return(im_stains)


def binarize_image(tile, im_nuclei_stain, foreground_threshold, local_radius_ratio=3, minimum_radius = 3):
    """
    Binarizes an image using a combination of local thresholding and bounding
    boxes to identify candidate locations of nuclei.

    Parameters
    ----------
    tile : numpy array
        RGB image.
    im_nuclei_stain : numpy array
        grayscale image.
    foreground_threshold : float
        threshold for separating foreground and background.
    local_radius_ratio : float, optional
        factor expand bounding box. The default is 3.
    minimum_radius : float, optional
        radius of smallest nuclei. The default is 3.
    """

    ## Apply initial global threshold
    img = cv2.cvtColor((im_nuclei_stain),cv2.COLOR_GRAY2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_flat = img_gray.flatten()
    thresh = np.round(threshold_otsu(img_gray_flat[img_gray_flat<foreground_threshold]))
    img_bin = np.copy(img_gray)
    img_bin[img_gray<thresh] = 255
    img_bin[img_gray>=thresh] = 0

    ## Fill small holes in the image
    img_bin = binary_fill_holes(img_bin.astype(bool))
    img_bin = img_bin.astype(np.uint8)

    ## Remove small structures in the image based on minimum_radius
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(minimum_radius,minimum_radius))
    opening = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN, kernel, iterations = 1)

    ## Identify connected regions("components") in the image
    regions = cv2.connectedComponents(opening)[1]
    obj_props = regionprops(regions, intensity_image=im_nuclei_stain)

    ## Initialize mask
    im_fgnd_mask = np.zeros(im_nuclei_stain.shape).astype(np.uint8)

    ## Iterate through regions found via global thresholding
    for obj in obj_props:

        # Skip thresholding on background component
        if (obj.label == 0):
            continue

        # Expand bounding box based on local_radius_ratio
        # The idea is to include more background for local thresholding.
        bbox = obj.bbox
        equivalent_diameter = obj.equivalent_diameter
        min_row = np.max([0, np.round(bbox[0] - equivalent_diameter*local_radius_ratio)]).astype(np.int)
        max_row = np.min([tile.shape[0], np.round(bbox[2] + equivalent_diameter*local_radius_ratio)]).astype(np.int)
        min_col = np.max([0, np.round(bbox[1] - equivalent_diameter*local_radius_ratio)]).astype(np.int)
        max_col = np.min([tile.shape[1], np.round(bbox[3] + equivalent_diameter*local_radius_ratio)]).astype(np.int)
        region = im_nuclei_stain[min_row:max_row, min_col:max_col]
        region_flat = region.flatten()

        # If local threshold fail. Default to global threshold instead.
        try:
            thresh = np.round(threshold_otsu(region_flat[region_flat<foreground_threshold]))
        except:
            thresh = foreground_threshold

        # Copy local bbox mask to larger tile mask
        region_bin = np.copy(region)
        region_bin[region<thresh] = 1
        region_bin[region>=thresh] = 0
        im_fgnd_mask[min_row:max_row, min_col:max_col] = im_fgnd_mask[min_row:max_row, min_col:max_col] + region_bin.astype(np.uint8)
        im_fgnd_mask[im_fgnd_mask>0] = 1

    return(im_fgnd_mask)


def find_nuclei(tile,im_nuclei_stain, im_fgnd_mask, min_nucleus_area=15):
    """
    Split the binary image of nuclei into a map of individual nuclei

    Parameters
    ----------
    tile : numpy array
        RGB image.
    im_nuclei_stain : numpy array
        grayscale image.
    im_fgnd_mask : numpy array
        binary mask of the foreground.
    min_nucleus_area : float, optional
        area of the smallest nuclei. The default is 15.
    """
    sure_fg_threshold = 0.50

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(im_fgnd_mask,cv2.MORPH_OPEN,kernel, iterations = 1)

    # Identify sure background area
    kernel = np.ones((5,5),np.uint8)
    sure_bg = cv2.dilate(opening,kernel,iterations=1)


    _ret, objects = cv2.connectedComponents(opening)
    obj_props = skimage.measure.regionprops(objects)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    stain_inverse = cv2.bitwise_not(im_nuclei_stain)
    stain_inverse = stain_inverse - np.min(stain_inverse[:])
    stain_inverse = (stain_inverse / np.max(stain_inverse[:])) * 255

    # Iterate through objects found
    sure_fg = np.zeros(im_nuclei_stain.shape)
    for obj in obj_props:
        bbox = obj.bbox

        # Calculate normalized distance map
        dist = dist_transform[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        dist = dist - np.min(dist[:])
        dist = (dist/np.max(dist[:]))*255

        # Normalize image region
        im = stain_inverse[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        im = im - np.min(im[:])
        im = (im/np.max(im[:]))*255

        # Combine distance and image then perform thresholding
        combined = im + dist
        _ret, temp = cv2.threshold(combined,sure_fg_threshold*np.max(combined[:]),255,0)

        # Save to sure foreground map
        sure_fg[bbox[0]:bbox[2], bbox[1]:bbox[3]] = temp


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==1] = 0

    markers = cv2.watershed(tile,markers)

    # Label boundary lines as background
    markers[markers==-1] = 1

    # Remove small objects according to min_nucleus area
    obj_props = skimage.measure.regionprops(markers)
    for obj in obj_props:
        if (obj.area < min_nucleus_area):
            markers[markers==obj.label] = 1

    obj_props = skimage.measure.regionprops(markers, intensity_image=im_nuclei_stain)
    return(markers, obj_props)
