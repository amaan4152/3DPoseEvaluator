import pandas as pd


def data_parse2(model_name: str, model_data: dict, joints: dict) -> pd.DataFrame:
    """
    Parse pose estimation data into DataFrames with the following order:
        [POSITION (XYZ), THETA (joint angle), QUATERNION (orientation), ... ]

    Parameters
    ----------
    model_name: name of pose estimation algo
    model_data: dictionary of estimated joint positions, joint angles, and orientations
    joints: dictionary of graphical relation of joints
            {'vertex' : [joint_a, vertex, joint_b] (any order)}

    Return
    ------
    DataFrame object of estimated data
    """
    df = None
    for v, joint_list in joints.items():
        # column names
        theta_column = [f"{model_name}:{v}:theta"]
        pos_columns = [
            f"{model_name}:{j}:pos-{axis}"
            for j in joint_list
            for axis in ["X", "Y", "Z"]
        ]
        quat_columns = [
            f"{model_name}:{j}:quat-{axis}"
            for j in joint_list
            for axis in ["W", "X", "Y", "Z"]
        ]

        # make dataframes
        df_theta = pd.DataFrame(model_data["theta"], columns=theta_column)

        # Note: need to make sure that the order of model_data matches column labels!
        df_joints = None
        for i in range(len(joint_list)):
            df_pos = pd.DataFrame(
                model_data["pos"][i],
                columns=pos_columns[(3 * i) : (3 * (i + 1))],
            )
            """
            df_quat = pd.DataFrame(
                model_data["quat"], columns=quat_columns[(4*i):(4*(i + 1))]
            )
            """
            df_joints = pd.concat([df_joints, df_pos], axis=1)

        df = pd.concat([df, df_theta, df_joints], axis=1)

    return df
