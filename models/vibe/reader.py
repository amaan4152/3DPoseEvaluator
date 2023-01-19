import joblib
import sys
from numpy import save


def main():
    """
    Parse VIBE pose data

    Return
    ------
    VIBE_data: array of 3D position of pose. The 1st dim is frame number
    (inclusive of 0), and the 2nd dim is the 3D position of all joints
    """
    pkl_file = sys.argv[1]
    save_dir = sys.argv[2]
    pkl_data = joblib.load(pkl_file)
    data = pkl_data[1]["joints3d"] * 1000
    save(f"{save_dir}/data.npy", data)

if __name__ == "__main__":
    main()
