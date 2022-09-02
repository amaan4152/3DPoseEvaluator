import joblib


def parse_vibe(pkl_file):
    """
    Parse VIBE pose data

    Parameters
    ----------
    pkl_file: pickle file of pose data in outputs folder after running VIBE

    Return
    ------
    VIBE_data: array of 3D position of pose. The 1st dim is frame number
    (inclusive of 0), and the 2nd dim is the 3D position of all joints
    """
    pkl_data = joblib.load(pkl_file)
    VIBE_data = pkl_data[1]["joints3d"] * 1000
    return VIBE_data
