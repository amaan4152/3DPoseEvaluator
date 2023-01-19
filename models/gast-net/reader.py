import sys
from numpy import load, array, save


def main():
    """
    Parse GAST pose data

    Return
    ------
    GAST_data: array of 3D position of pose. The 1st dim is frame number
    (inclusive of 0), and the 2nd dim is the 3D position of all joints
    """
    # WARNING: 3D coordinate system for GAST revolves around pelvis (keypoint #0) as origin
    # 3D array: dim_03->list of frame data of vid seq; dim_02 -> indexed by # of frames; dim_01 -> indexed by keypoint #
    npz_file = sys.argv[1]
    save_dir = sys.argv[2]
    GAST_data = load(npz_file)["reconstruction"]
    data = array(GAST_data[0]) * 1000  # convert data to millimeters
    save(f"{save_dir}/data.npy", data)

if __name__ == "__main__":
    main()
