#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Roland Ringgenberg
# DATE CREATED: 11/24/2025                                 
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames
    of the image files. These pet image labels are used to check the accuracy
    of the labels that are returned by the classifier function, since the
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')

    Parameters:
        image_dir (str): The (full) path to the folder of images that are to be
            classified by the classifier function.

    Returns:
        dict: Dictionary with 'key' as image filename and 'value' as a list.
            The list contains the following item:
                index 0 = pet image label (string)
    """

    # Create results dictionary
    results_dic = dict()

    # Retrieve the filenames from the image directory
    try:
        filenames = listdir(image_dir)
    except Exception as e:
        # Propagate a clear error if directory cannot be listed
        raise RuntimeError(f"Unable to list directory '{image_dir}': {e}")

    for fname in filenames:
        # Skip hidden/system files
        if fname.startswith('.'):
            continue

        # Generate pet label from filename:
        # - lowercase
        # - split by underscores
        # - keep only alphabetic tokens
        # - join with spaces and strip
        lower_name = fname.lower()
        words = lower_name.split('_')
        label_words = [w for w in words if w.isalpha()]
        pet_label = " ".join(label_words).strip()

        if fname not in results_dic:
            results_dic[fname] = [pet_label]
        else:
            # Warn about duplicate filenames (shouldn't happen in a normal dataset)
            print(f"Warning: Duplicate filename encountered and ignored: {fname}")

    return results_dic
