import numpy as np
from feature_algorithms.colorvariability import colorvariability
from feature_algorithms.bluewhiteveil import evaluate_blue_white_veil
from feature_algorithms.streaks import detection
from feature_algorithms.asymmetry import asymmetry
from feature_algorithms.dotsglobules import dots_and_globules


#Main function to extract features from an image, that calls other functions    
def extract_features(image, mask):
    try:
        colorvariabilityfeature = colorvariability(image, mask)
    except Exception as e:
        colorvariabilityfeature = None
        print(f"Skipping color variability due to error: {e}")
        return np.array([0.5, 0.5, 0.5], dtype=np.float16)

    try:
        bluewhiteveilfeature = evaluate_blue_white_veil(image, mask)
    except Exception as e:
        bluewhiteveilfeature = None
        print(f"Skipping blue-white veil detection due to error: {e}")
        return np.array([0.5, 0.5, 0.5], dtype=np.float16)

    try:
        streaksfeature = detection(image, mask)
    except Exception as e:
        streaksfeature = None
        print(f"Skipping streaks detection due to error: {e}")
        return np.array([0.5, 0.5, 0.5], dtype=np.float16)

    try:
        asymmetryfeature = asymmetry(mask)
    except Exception as e:
        asymmetryfeature = None
        print(f"Skipping asymmetry calculation due to error: {e}")
        return np.array([0.5, 0.5, 0.5], dtype=np.float16)

    try:
        dotsglubesfeature = dots_and_globules(image, mask)
    except Exception as e:
        dotsglubesfeature = None
        print(f"Skipping dots and globules detection due to error: {e}")
        return np.array([0.5, 0.5, 0.5], dtype=np.float16)

    return np.array([colorvariabilityfeature, streaksfeature, asymmetryfeature], dtype=np.float16)
    
        







