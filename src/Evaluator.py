import numpy as np
import pandas as pd
from tqdm import trange
from typing import Tuple


class Evaluator(object):
    MOD_FPS = 60
    GND_FPS = 120
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(self, model: str, video: str):
        self.model_name = model
        df_model = pd.read_csv(
            f"output/{model}/{model}-{video}-raw_data.csv", header=[0, 1, 2, 3]
        )
        self.df_theta_model = df_model.iloc[:, 1]
        self.df_pos_model = df_model.iloc[:, 2:]
        self.df_gnd = pd.read_csv(
            f"output/gnd_truth/gnd-{video}-raw_data.csv", header=[0, 1, 2, 3]
        ).iloc[:, 1:]
        self.__gnd_align()
        self.df_theta_gnd = self.df_gnd.iloc[:, 0].to_frame()
        self.df_pos_gnd = self.df_gnd.iloc[:, 1:]

    def calibrate(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        d = 3  # dimensionality of data
        n = self.df_pos_model.columns.size // d  # number of joints
        num_frames = self.df_pos_model.shape[0]  # number of frames
        assert (
            num_frames % N == 0
        ), f"[{self.FAIL}ERROR{self.ENDC}]: {N=} must be multiple of {num_frames=}"

        # apply optimal rotation and translation on model data
        transform = lambda m: (self.R @ m) + self.t

        # split along column axis into `n` groups
        hsplit = lambda arr: arr.reshape(arr.shape[0], n, d).swapaxes(0, 1)

        # split along row axis into `n` groups
        vsplit = (
            lambda arr: arr.reshape(d, n, arr.shape[1] // n)
            .swapaxes(0, 1)
            .transpose(0, 2, 1)
        )

        # calibrate group of N frames
        m_pos_t = np.zeros(shape=self.df_pos_model.shape)
        m_pos_data = self.df_pos_model.values
        g_pos_data = self.df_pos_gnd.values

        # compute R,t
        m_pos_samples = np.vstack(hsplit(m_pos_data[0:N, :]))
        g_pos_samples = np.vstack(hsplit(g_pos_data[0:N, :]))
        self.__arun(m_pos_samples.T, g_pos_samples.T)
        for i in np.arange(start=N, stop=num_frames, step=N):
            m_pos_t[i : (N + i), :] = np.hstack(
                vsplit(transform(m_pos_samples.T))
            )  # shape = (N, d*n)
            m_pos_samples = np.vstack(hsplit(m_pos_data[i : (N + i), :]))

        MCAL_pos = hsplit(m_pos_t)
        return MCAL_pos, self.df_theta_model.values

    def calibration_metric(
        self, window_size: int, num_experiments: int, sample_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        out_list = [None] * num_experiments
        for i in trange(num_experiments):
            cal_pos_data = self.calibrate(N=window_size)[0]
            num_rows = cal_pos_data.shape[1]
            if i == 0:
                inds = np.random.choice(num_rows, sample_size)
            pos_set = np.hstack([data[inds, :] for data in cal_pos_data])
            out_list[i] = pos_set

        out_data = pd.DataFrame(np.vstack(out_list))
        out_data.columns = self.df_pos_model.columns
        return out_data, out_data.describe()

    def MPJPE(self, df_pos_cal: pd.DataFrame):
        # retrieve joint sets
        cal_joint_sets = self.__get_joint_sets(df_pos_cal)
        gnd_joint_sets = self.__get_joint_sets(self.df_pos_gnd)

        # compute error distance between GND and MODEL
        err = cal_joint_sets - gnd_joint_sets
        dist = np.linalg.norm(err, axis=2)
        mean_dist = dist.mean(axis=1)
        return mean_dist

    def PDJ(self, df_pos_cal: pd.DataFrame, torso_diam: float):
        # retrieve joints
        cal_joint_sets = self.__get_joint_sets(df_pos_cal)
        gnd_joint_sets = self.__get_joint_sets(self.df_pos_gnd)

        # compute error distance between GND and MODEL
        err = cal_joint_sets - gnd_joint_sets
        dist = np.linalg.norm(err, axis=2)  # shape = (J, N)

        # check threshold
        num_detected = (dist < (0.2 * torso_diam)).astype(int)
        PDJ = num_detected.mean(axis=1)
        return PDJ

    def __arun(self, A: np.ndarray, B: np.ndarray) -> None:
        shape = A.shape  # shape=(d,n)

        # compute centroids
        Ac = A.T.mean(axis=0).reshape((-1, 1))
        Bc = B.T.mean(axis=0).reshape((-1, 1))

        # normalize to center
        A = A - np.tile(Ac, (1, shape[1]))
        B = B - np.tile(Bc, (1, shape[1]))

        H = A @ B.T  # outer product
        U, S, V = np.linalg.svd(H)
        V = V.T
        self.R = V @ U.T  # rotation marix
        self.t = Bc - (self.R @ Ac)  # translation matrix

    def __gnd_align(self) -> None:
        step = int(self.GND_FPS / self.MOD_FPS)
        self.df_gnd = self.df_gnd.iloc[::step, :].reset_index(drop=True)
        delay = self.df_gnd.shape[0] - self.df_theta_model.shape[0] - 1
        self.df_gnd = self.df_gnd.iloc[0 : (-1 - delay), :].reset_index(drop=True)

    def __get_joint_sets(self, df: pd.DataFrame) -> np.ndarray:
        # split along column axis into `n` groups
        d = 3  # dimensionality
        n = self.df_pos_model.columns.size // d  # number of joints
        hsplit = lambda arr: arr.reshape(arr.shape[0], n, d).swapaxes(0, 1)
        return hsplit(df.values)
