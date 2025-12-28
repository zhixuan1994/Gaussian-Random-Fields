from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import math
from tqdm import tqdm
from mpmath import mp, mpf, gamma
from scipy.special import lambertw

# Reference: Peng and Wu (2025), with the python package fractal-analysis 0.3.5, 
# and the link is https://pypi.org/project/fractal-analysis/

class WoodChanFgnSimulator:
    def __init__(self, sample_size: int, hurst_parameter: float, tmax: float = 1, std_const: float = 1):
        self.sample_size = sample_size
        if self.sample_size <= 0:
            raise ValueError(f'sample_size must be a positive integer.')
        self.hurst_parameter = hurst_parameter
        if not (0 < self.hurst_parameter < 1):
            raise ValueError(f'hurst_parameter must be in range (0, 1).')
        self.tmax = tmax
        if not self.tmax > 0:
            raise ValueError(f'tmax must be positive.')
        self.std_const = std_const
        if not self.std_const > 0:
            raise ValueError(f'std_const must be positive.')

    def _first_line_circulant_matrix(self, m, cov: Callable, prev_k=None, prev_v=None):
        new_k = self.tmax * np.arange(0, m / 2 + 1, dtype=int)

        if prev_k is not None and prev_v is not None:
            prev_len = len(prev_k)
            if prev_len >= len(new_k):
                v = prev_v[:len(new_k)]
            else:
                extra_k = new_k[prev_len:]
                extra_v = cov(k=extra_k)
                v = np.concatenate((prev_v, extra_v))
        else:
            v = cov(k=new_k)

        ind = np.concatenate((np.arange(0, m / 2, dtype=int), np.arange(m / 2, 0, -1, dtype=int)))
        line = v[ind]
        return line, new_k, v

    @staticmethod
    def _simulate_w(m, seed: int = None):
        np.random.seed(seed)
        ar = np.random.normal(0, 1, int(m / 2 + 1))
        ai = np.random.normal(0, 1, int(m / 2 + 1))
        ar[0] = 2 ** 0.5 * ar[0]
        ar[-1] = 2 ** 0.5 * ar[-1]
        ai[0] = 0
        ai[-1] = 0
        ar = np.concatenate((ar, ar[int(m / 2 - 1): 0:-1]))
        aic = -ai
        ai = np.concatenate((ai, aic[int(m / 2 - 1): 0:-1]))
        w = [complex(one_ar, one_ai) for one_ar, one_ai in zip(ar, ai)]
        return w

    def get_fgn(self, cov: Callable, N: int, seed: int = None, is_precise: bool = False) -> np.ndarray:
        m = 2 ** (int(math.log(N - 1) / math.log(2) + 1))
        eigc, k_vals, v_vals = self._first_line_circulant_matrix(m=m, cov=cov)
        eigc = fft(eigc)
        if not is_precise:
            eigc = np.clip(eigc, 1e-10, None)
        else:
            while any(v <= 0 for v in eigc) and m < 2 ** 17:
                m = 2 * m
                eigc, k_vals, v_vals = self._first_line_circulant_matrix(m=m, cov=cov, prev_k=k_vals, prev_v=v_vals)
                eigc = fft(eigc).real
        w = self._simulate_w(m=m, seed=seed)
        # reconstruction of the fgn
        w = np.sqrt(eigc.astype(np.cdouble)) * w
        fgn = fft(w)
        fgn = fgn / (2 * m) ** 0.5
        fgn = fgn.real
        return fgn

    def plot(self, series: np.ndarray, method_name: str, series_name: str, save_path: str = None,
             y_limits: list = None):
        plt.plot(np.arange(0, self.tmax, self.tmax / len(series)), series)
        plt.title(
            f'{method_name} {series_name} simulation with {len(series)} samples and {self.hurst_parameter} hurst')
        plt.xlabel('Time')
        if y_limits is not None:
            plt.ylim(y_limits)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class DprwSelfSimilarFractalSimulator(WoodChanFgnSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, covariance_func: Callable,
                 lamperti_multiplier: int = 5, factor: float = None, tmax: float = 1, std_const: float = 1):
        super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter, tmax=tmax, std_const=std_const)
        self.covariance_func = covariance_func
        if not isinstance(lamperti_multiplier, int) and lamperti_multiplier <= 0:
            raise ValueError(f'lamperti_multiplier must be a positive integer.')
        self.lamperti_multiplier = lamperti_multiplier
        self.lamperti_series_len = self.lamperti_multiplier * self.sample_size
        self.factor = factor

    @property
    def _lamperti_subseq_index(self):
        seires_step = self.tmax / self.sample_size
        series_t = np.arange(start=seires_step, stop=self.tmax + seires_step, step=seires_step)
        log_series_t = np.log(series_t) + np.abs(np.log(series_t[0]))
        max_log_series_exp_t = np.max(log_series_t)
        lamperti_subseq_index = np.rint(log_series_t * self.lamperti_series_len / max_log_series_exp_t) - 1
        lamperti_subseq_index[0] = 0
        return series_t, lamperti_subseq_index.astype(int)

    def get_self_similar_process(self, is_plot=False, method_name=None, series_name=None, seed=None,
                                 plot_path: str = None, y_limits: list = None):
        series_t, lamperti_subseq_index = self._lamperti_subseq_index
        lamp_fgn = self.get_fgn(seed=seed, N=self.lamperti_series_len, cov=self.covariance_line)
        lamp_fgn = lamp_fgn - lamp_fgn[0]
        return lamp_fgn, lamperti_subseq_index

    def covariance_with_adaptive_precision(self, k_de, n_de, hurst_de, factor_de, tolerance=0.0001, initial_prec=17,
                                           step=5, max_prec=1000000):
        mp.dps = initial_prec

        tolerance = mpf(str(tolerance))
        v_prev = mpf('0')
        v = self.covariance_func(k_de=k_de, n_de=n_de, hurst_de=hurst_de, factor_de=factor_de)
        return float(v)
        

    def covariance_line(self, k):
        n_de = mpf(str(self.sample_size))
        hurst_de = mpf(str(self.hurst_parameter))

        if self.factor is not None:
            factor_de = mpf(str(self.factor))
        else:
            factor_de = None
        v = np.array([
            self.covariance_with_adaptive_precision(k_de=mpf(str(k_ele)), n_de=n_de, hurst_de=hurst_de,
                                                    factor_de=factor_de) for
            k_ele in k])
        return v

class DprwBiFbmSimulator(DprwSelfSimilarFractalSimulator):
    def __init__(self, sample_size: int, hurst_parameter: float, 
                 lamperti_multiplier: int = 5,
                 tmax: float = 1, std_const: float = 1,
                 FBM_cov_md: int = 1, bi_factor: float=0.7):

        self.bi_factor = bi_factor
        if FBM_cov_md == 1:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            covariance_func=self.fbm_cov, factor=self.bi_factor,
                            lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)
        elif FBM_cov_md == 2:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            covariance_func=self.sub_fbm_cov, factor=self.bi_factor,
                            lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)
        elif FBM_cov_md == 3:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            covariance_func=self.bi_fbm_cov, factor=self.bi_factor,
                            lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)
        elif FBM_cov_md == 4:
            super().__init__(sample_size=sample_size, hurst_parameter=hurst_parameter,
                            covariance_func=self.tri_fbm_cov, factor=self.bi_factor,
                            lamperti_multiplier=lamperti_multiplier, tmax=tmax, std_const=std_const)
    
    def fbm_cov(self, k_de, n_de, hurst_de, factor_de):
        N = n_de
        t_s = k_de/N
        s_t = -t_s
        H = hurst_de

        temp_1 = np.abs(N**(t_s/2) - N**(s_t/2))**(2*H)
        temp_2 = np.abs(N**((t_s-1/N)/2) - N**((s_t+1/N)/2))**(2*H)
        temp_3 = np.abs(N**((t_s+1/N)/2) - N**((s_t-1/N)/2))**(2*H)
        abs_res = - temp_1 + (temp_2 + temp_3)/2
        
        new_temp_1 = N**(t_s*H) + N**(s_t*H)
        new_temp_2 = N**((t_s-1/N)*H) + N**((s_t+1/N)*H)
        new_temp_3 = N**((t_s+1/N)*H) + N**((s_t-1/N)*H)
        no_abs_res = new_temp_1 - (new_temp_2 + new_temp_3)/2
        return abs_res + no_abs_res
    
    def C_H(self, H):
        if np.abs(H - 0.5) < 0.01:
            return np.pi
        else:
            return gamma(2-2*H) / (H*(1-2*H))
    
    def sub_fbm_cov(self, k_de, n_de, hurst_de, factor_de):
        N = n_de
        t_s = k_de/N
        s_t = -t_s
        H = float(hurst_de)

        CH = self.C_H(H)
        gam_2H = gamma(2*H)
        CH_2 = np.sqrt(np.pi / (2*H * gam_2H * np.sin(np.pi*H)))
        CH_2 = np.float64(CH_2)
        temp_cons = CH/CH_2**2

        temp_1 = np.abs(N**((t_s + 1/N)/2) - N**((s_t + 1/N)/2))**(2*H)
        temp_2 = np.abs(N**((t_s - 1/N)/2) - N**((s_t + 1/N)/2))**(2*H)
        temp_3 = np.abs(N**((t_s + 1/N)/2) - N**((s_t - 1/N)/2))**(2*H)
        temp_4 = np.abs(N**(t_s/2) - N**(s_t/2))**(2*H)
        abs_res = (-temp_1 + temp_2 + temp_3 - temp_4)/2

        new_temp_1 = N**((s_t + 1/N)*H) + N**((t_s + 1/N)*H) - (N**((t_s + 1/N)/2) + N**((s_t + 1/N)/2))**(2*H)/2
        new_temp_2 = N**((t_s - 1/N)*H) + N**((s_t + 1/N)*H) - (N**((t_s - 1/N)/2) + N**((s_t + 1/N)/2))**(2*H)/2
        new_temp_3 = N**((t_s + 1/N)*H) + N**((s_t - 1/N)*H) - (N**((t_s + 1/N)/2) + N**((s_t - 1/N)/2))**(2*H)/2
        new_temp_4 = N**(t_s*H) + N**(s_t*H) - (N**(t_s/2) + N**(s_t/2))**(2*H)/2
        non_abs_res = new_temp_1 - new_temp_2 - new_temp_3 + new_temp_4
        return (non_abs_res + abs_res)* temp_cons
    
    def bi_fbm_cov(self, k_de, n_de, hurst_de, factor_de):
        N = n_de
        t_s = k_de/N
        s_t = -t_s
        H = float(hurst_de)
        K = factor_de
        
        temp_1 = np.abs(N**(t_s/2) - N**(s_t/2))**(2*H*K)
        temp_2 = np.abs(N**((t_s - 1/N)/2) - N**((s_t + 1/N)/2))**(2*H*K)
        temp_3 = np.abs(N**((t_s + 1/N)/2) - N**((s_t - 1/N)/2))**(2*H*K)
        temp_4 = np.abs(N**(t_s/2) - N**(s_t/2))**(2*H*K)
        abs_res = -temp_1 + temp_2 + temp_3 - temp_4

        new_temp_1 = (N**(t_s*H) + N**(s_t*H))**K
        new_temp_2 = (N**((t_s - 1/N)*H) + N**((s_t + 1/N)*H))**K
        new_temp_3 = (N**((t_s + 1/N)*H) + N**((s_t - 1/N)*H))**K
        new_temp_4 = (N**(t_s*H) + N**(s_t*H))**K
        non_abs_res = new_temp_1 - new_temp_2 - new_temp_3 + new_temp_4
        return (non_abs_res + abs_res) / 2**K
    
    def tri_fbm_cov(self, k_de, n_de, hurst_de, factor_de):
        N = n_de
        t_s = k_de/N
        s_t = -t_s
        H = float(hurst_de)
        K = factor_de

        temp1 = N**(H*K*t_s) + N**(H*K*s_t) - (N**(t_s*H) + N**(s_t*H))**K
        temp2 = N**((t_s - 1/N)*H*K) + N**((s_t + 1/N)*H*K) - (N**((t_s - 1/N)*H) + N**((s_t + 1/N)*H))**K
        temp3 = N**((t_s + 1/N)*H*K) + N**((s_t - 1/N)*H*K) - (N**((t_s + 1/N)*H) + N**((s_t - 1/N)*H))**K
        temp4 = N**(t_s*H*K) + N**(s_t*H*K) - (N**(t_s*H) + N**(s_t*H))**K
        return temp1 - temp2 - temp3 + temp4

    def get_fbm(self, is_plot=False, seed=None, plot_path: str = None, y_limits: list = None):
        bi_fbm = self.get_self_similar_process(is_plot=is_plot, seed=seed, method_name='DPRW',
                                               series_name=f'{self.bi_factor} Bi-FBM', plot_path=plot_path,
                                               y_limits=y_limits)
        return bi_fbm
    
def scaling_subseq_index(fgn_size, sample_size):
        seires_step = 1 / fgn_size
        series_t = np.arange(start=seires_step, stop=1+ seires_step, step=seires_step)[:sample_size]
        scaling_subseq_index = np.floor(
            (np.log(series_t) / np.log(fgn_size) + 1) * fgn_size)-1
        scaling_subseq_index[0] = 0
        return scaling_subseq_index.astype(int)

def fbm_main(sample_size, case = 1, cov_md = 1):
    H_list = np.arange(0, 1, 1/100)[1:]
    
    if case ==1 :
        # Case 1
        H_list = np.ones(len(H_list))*0.2
        H_list[int(len(H_list)/2):] = 0.8
        # Case 2
    elif case == 2:
        H_list = 0.2 + 0.6*H_list
        # Case 3
    elif case == 3:
        H_list = 0.25 + 2*(H_list - 0.5)**2
    elif case == 4:
        # Case 4
        H_list = 0.5 + 0.3 * np.cos(H_list * np.pi * 6)
    sample_size_new = np.ceil(np.real(lambertw(-np.log(1/(sample_size-1)+1), k=-1))/(-np.log(1/(sample_size-1)+1)))

    X_ma_ini, Lag_series = DprwBiFbmSimulator(sample_size=sample_size_new, hurst_parameter=H_list[0], FBM_cov_md=cov_md).get_fbm()
    X_matrix = [X_ma_ini]
    for H in tqdm(H_list[1:]):
        path_H_k = DprwBiFbmSimulator(sample_size=sample_size_new, hurst_parameter=H, FBM_cov_md=cov_md).get_fbm()[0]
        X_matrix.append(path_H_k)
    X_ma = np.asarray(X_matrix)
    siqu = scaling_subseq_index(sample_size_new, sample_size)

    return X_ma[:, Lag_series][:, siqu], H_list