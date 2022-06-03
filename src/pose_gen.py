from model_analysis import get_poseData2
from data_parser import data_parse2


def pose_gen(video, ots_file, model_type, animation, sframe, fframe, joints, duration):
    # DO NOT NEED FRAME_INFO PARAMETER -> NEED TO FIX get_PoseData!
    #   blazepose function requires frame_info to find total # of frames
    '''
    OTS_pos, OTS_quat, OTS_theta, skpd_frames, torso_width, frame_stat = get_OTSData(
        ots_file, sframe, fframe
    )
    ots_data = {"theta": OTS_theta, "pos": OTS_pos, "quat": OTS_quat}
    df_ots_raw = data_parse2("OTS", ots_data, joints)
    '''
    MODEL_pos, MODEL_theta, MODEL_quat = get_poseData2(video, model_type, animation)
    model_data = {"theta": MODEL_theta, "pos": MODEL_pos, "quat": MODEL_quat}

    df_m_raw = data_parse2(model_type, model_data, joints)  # raw pose data
    # df_raw = pd.concat([df_ots_raw, df_m_raw], axis=1)

    return df_m_raw
