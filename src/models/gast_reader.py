from numpy import load, array


def parse_gast(npz_file):
    """
    Parse GAST pose data

    Parameters
    ----------
    npz_file: numpy zipped file of pose data in outputs folder after running GAST

    Return
    ------
    GAST_data: array of 3D position of pose. The 1st dim is frame number
    (inclusive of 0), and the 2nd dim is the 3D position of all joints
    """
    # WARNING: 3D coordinate system for GAST revolves around pelvis (keypoint #0) as origin
    # 3D array: dim_03->list of frame data of vid seq; dim_02 -> indexed by # of frames; dim_01 -> indexed by keypoint #
    GAST_data = load(npz_file)["reconstruction"]
    GAST_data = array(GAST_data[0]) * 1000  # convert data to millimeters

    return GAST_data
