def parse_vibe(pkl_data):
    """
    Parse VIBE pose data

    Parameters
    ----------
    pkl_data: pickle file of pose data in outputs folder after running VIBE

    Return
    ------
    VIBE_data: dictionary of 3D position of pose. The key is frame number
    (inclusive of 0), and the value is the 3D position of all joints
    """
    VIBE_data = {
        str(k): [v[9] * 1000, v[10] * 1000, v[11] * 1000]
        for k, v in zip(
            range(0, len(pkl_data[1]["joints3d"])), pkl_data[1]["joints3d"]
        )
    }

    return VIBE_data