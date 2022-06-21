from typing import Tuple
import numpy as np
import pandas as pd
import scipy

class Evaluator:
    MOD_FPS = 60
    GND_FPS = 120
    FAIL = "\033[91m"
    ENDC = "\033[0m"

    def __init__(self, model: str, video: str):
        df_model = pd.read_csv(
            f"output/{model.lower()}-{video}-raw_data.csv", header=[0,1,2,3]
        )
        self.df_theta_model = df_model.iloc[:, 1]
        self.df_pos_model = df_model.iloc[:, 2:]
        self.df_gnd = pd.read_csv(f"output/gnd-{video}-raw_data.csv", header=[0,1,2,3]).iloc[:, 1:]
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
        vsplit = lambda arr: arr.reshape(d, n, arr.shape[1] // n).swapaxes(0, 1).transpose(0, 2, 1)

        # calibrate group of N frames
        m_pos_t = np.zeros(shape=self.df_pos_model.shape)
        m_pos_data = self.df_pos_model.values
        g_pos_data = self.df_pos_gnd.values
        for i in np.arange(start=0, stop=num_frames, step=N):
            m_pos_samples = np.vstack(hsplit(m_pos_data[i : (N + i), :]))
            g_pos_samples = np.vstack(hsplit(g_pos_data[i : (N + i), :]))
            self.__arun(m_pos_samples.T, g_pos_samples.T)
            m_pos_t[i : (N + i), :] = np.hstack(vsplit(transform(m_pos_samples.T)))  # shape = (N, d*n)

        MCAL_pos = hsplit(m_pos_t)
        return MCAL_pos, self.df_theta_model.values

    def __arun(self, A : np.ndarray, B : np.ndarray) -> None:
        shape = A.shape  # shape=(d,n)

        Ac = A.T.mean(axis=0).reshape((-1, 1))
        Bc = B.T.mean(axis=0).reshape((-1, 1))

        A = A - np.tile(Ac, (1, shape[1]))
        B = B - np.tile(Bc, (1, shape[1]))

        H = A @ B.T
        U, S, V = np.linalg.svd(H)
        V = V.T
        self.R = V @ U.T
        self.t = Bc - (self.R @ Ac)

    def __gnd_align(self) -> None:
        step = int(self.GND_FPS / self.MOD_FPS)
        self.df_gnd = self.df_gnd.iloc[::step, :].reset_index(drop=True)
        delay = self.df_gnd.shape[0] - self.df_theta_model.shape[0] - 1
        self.df_gnd = self.df_gnd.iloc[0 : (-1 - delay), :].reset_index(
            drop=True
        )