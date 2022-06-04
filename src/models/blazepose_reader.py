from numpy import load, array
def parse_blazepose(npz_file):
    """
    Parse Blazepose pose data

    Parameters
    ----------
    npz_file: numpy zipped file of pose data in outputs folder after running Blazepose

    Return
    ------
    GAST_data: array of 3D position of pose. The 1st dim is frame number
    (inclusive of 0), and the 2nd dim is the 3D position of all joints
    """
    # WARNING: 3D coordinate system for GAST revolves around pelvis (keypoint #0) as origin
    # 3D array: dim_03->list of frame data of vid seq; dim_02 -> indexed by # of frames; dim_01 -> indexed by keypoint #
    BP_data = load(npz_file)["pose_data"]
    return BP_data