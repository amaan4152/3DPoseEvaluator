import sys
from numpy import load, save


def main():
    """
    Parse Blazepose pose data

    Parameters
    ----------
    npz_file: numpy zipped file of pose data in outputs folder after running GAST
    """
    # WARNING: 3D coordinate system for GAST revolves around pelvis (keypoint #0) as origin
    # 3D array: dim_03->list of frame data of vid seq; dim_02 -> indexed by # of frames; dim_01 -> indexed by keypoint #
    npz_file = sys.argv[1]
    save_dir = sys.argv[2]
    data = load(npz_file)["pose_data"] * 1000
    save(f"{save_dir}/data.npy", data)

if __name__ == "__main__":
    main()