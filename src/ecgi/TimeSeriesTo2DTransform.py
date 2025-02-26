import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from scipy.spatial.distance import pdist, squareform
from scipy.signal import resample, windows
from scipy.signal import resample
import scipy.signal as signal

class TimeSeriesTo2DTransform:
    """
    Clase para transformar series temporales 1D a 2D (RPM, GAF, MTF, RP, JRP, STFT, CWT, DWT, etc.)
    """

    def __init__(self, fs: float = 1.0):
        self.fs = fs
        self.transformations = {
            'rpm': self.relative_position_matrix,
            'gaf': self.gramian_angular_field,
            'gaf_diff': self.gramian_angular_difference_field,
            'mtf': self.markov_transition_field,
            'rp': self.recurrence_plot,
            'jrp': self.joint_recurrence_plot,
            'spectrogram': self.spectrogram,
            'cwt': self.cwt,
            'dwt': self.dwt,
            'stft': self.stft
        }

    def transform(self,
                  x: torch.Tensor,
                  method: str,
                  paa_factor: int = 1,
                  return_numpy: bool = False,
                  **kwargs) -> Union[torch.Tensor, np.ndarray]:
        """
        Aplica la transformación 'method' sobre x (B,C,T).
        Retorna un tensor 4D.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if method not in self.transformations:
            raise ValueError(f"Método no soportado: {method}.")

        # Z-score
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean)/std

        # PAA
        if paa_factor>1:
            x_norm = self.piecewise_aggregate(x_norm, paa_factor)

        transform_func = self.transformations[method]
        B, C, _ = x_norm.shape
        out_list = []
        for b in range(B):
            channel_outs = []
            for c in range(C):
                out_2d = transform_func(x_norm[b,c], **kwargs)
                channel_outs.append(out_2d)

            if method in ['cwt','dwt']:
                # => (n_scales, time, channels)
                out_tensor = torch.stack(channel_outs, dim=-1)
                out_list.append(out_tensor)
            else:
                # => (channels, dim1, dim2)
                out_tensor = torch.stack(channel_outs)
                out_list.append(out_tensor)

        if method in ['cwt','dwt']:
            result = torch.stack(out_list, dim=0)  # (B, n_scales, time, channels)
        else:
            result = torch.stack(out_list)        # (B, channels, dim1, dim2)

        if return_numpy:
            return result.detach().cpu().numpy()
        return result

    def piecewise_aggregate(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        b, c, t = x.shape
        new_len = t//factor
        r = t%factor
        if r>0:
            pad_size=factor-r
            x = F.pad(x, (0,pad_size))
            t+=pad_size
        x_reshaped = x.reshape(b,c,new_len,factor)
        return x_reshaped.mean(dim=3)

    def min_max_scaler(self, x: torch.Tensor, feature_range=(0,1)) -> torch.Tensor:
        mn, mx = x.min(), x.max()
        if mx>mn:
            scaled = (x - mn)/(mx - mn)
            scaled = scaled*(feature_range[1]-feature_range[0]) + feature_range[0]
            return scaled
        else:
            return torch.ones_like(x)*((feature_range[0]+feature_range[1])/2)

    # -----------------------
    # Métodos concretos
    # -----------------------

    def relative_position_matrix(self, x: torch.Tensor) -> torch.Tensor:
        x_r = x.unsqueeze(0)
        x_c = x.unsqueeze(1)
        diff_ = x_r - x_c
        return self.min_max_scaler(diff_)

# En TimeSeriesTo2DTransform.py (dentro de gramian_angular_field):

    # En TimeSeriesTo2DTransform:
    def gramian_angular_field(self, x: torch.Tensor, gaf_mode: str = 'summation') -> torch.Tensor:
        """
        Realiza Gramian Angular Field. 'gaf_mode' = 'summation' o 'difference'.
        """
        x_norm = self.min_max_scaler(x, (-1,1))
        phi = torch.acos(x_norm)
        cos_, sin_ = torch.cos(phi), torch.sin(phi)

        if gaf_mode == 'summation':
            # GASF
            result = cos_.unsqueeze(0)*cos_.unsqueeze(1) - sin_.unsqueeze(0)*sin_.unsqueeze(1)
        elif gaf_mode == 'difference':
            # GADF
            result = sin_.unsqueeze(0)*cos_.unsqueeze(1) - cos_.unsqueeze(0)*sin_.unsqueeze(1)
        else:
            raise ValueError("gaf_mode must be 'summation' or 'difference'")

        return self.min_max_scaler(result)



    def gramian_angular_difference_field(self, x: torch.Tensor) -> torch.Tensor:
        return self.gramian_angular_field(x, method='difference')

    def markov_transition_field(self, x: torch.Tensor, n_bins=8) -> torch.Tensor:
        x_np = x.cpu().numpy()
        t = len(x_np)
        bins = np.linspace(x_np.min(), x_np.max(), n_bins+1)
        quant = np.digitize(x_np, bins)-1
        quant = np.clip(quant, 0, n_bins-1)
        trans_mat = np.zeros((n_bins,n_bins), dtype=np.float32)
        for i in range(t-1):
            trans_mat[quant[i], quant[i+1]]+=1
        row_sum = trans_mat.sum(axis=1, keepdims=True)
        row_sum[row_sum==0]=1
        trans_mat/=row_sum
        mtf = np.zeros((t,t), dtype=np.float32)
        for i in range(t):
            for j in range(t):
                mtf[i,j] = trans_mat[quant[i], quant[j]]
        return self.min_max_scaler(torch.tensor(mtf))

    def recurrence_plot(self, x: torch.Tensor, threshold=0.2, metric='euclidean') -> torch.Tensor:
        x_np = x.cpu().numpy().reshape(-1,1)
        dist_mat = squareform(pdist(x_np, metric=metric))
        rp = (dist_mat<threshold).astype(np.float32)
        return torch.tensor(rp)

    def joint_recurrence_plot(self, x: torch.Tensor, embedding_dim=3, time_delay=1, threshold=0.1) -> torch.Tensor:
        x_np = x.cpu().numpy()
        length=len(x_np)
        emb_len= length - (embedding_dim-1)*time_delay
        if emb_len<2:
            raise ValueError("Embedding demasiado grande para jrp.")
        embed = np.zeros((emb_len, embedding_dim), dtype=np.float32)
        for i in range(emb_len):
            for j in range(embedding_dim):
                embed[i,j] = x_np[i + j*time_delay]
        dist_ = squareform(pdist(embed,'euclidean'))
        jrp = (dist_<threshold).astype(np.float32)
        return torch.tensor(jrp)

    def spectrogram(self, x: torch.Tensor, nperseg=32, noverlap=None) -> torch.Tensor:

        x_np = x.cpu().numpy()
        if nperseg>len(x_np)//2:
            nperseg=len(x_np)//2
        if noverlap is None:
            noverlap=int(nperseg*0.75)
        freqs, times, Sxx = signal.spectrogram(
            x_np, fs=self.fs, window='hann',
            nperseg=nperseg, noverlap=noverlap,
            detrend=False, scaling='spectrum'
        )
        log_ = np.log1p(Sxx)
        spec_t = torch.tensor(log_, dtype=torch.float32)
        return self.min_max_scaler(spec_t)

    def stft(self, x: torch.Tensor, window_type='hann', nperseg=256, noverlap=None) -> torch.Tensor:
        import scipy.signal as signal
        x_np = x.cpu().numpy()
        if nperseg>len(x_np)//2:
            nperseg=len(x_np)//2
        if noverlap is None:
            noverlap=int(nperseg*0.75)
        if window_type.lower()=='hann':
            window='hann'
        elif window_type.lower()=='blackman':
            window='blackman'
        elif window_type.lower()=='kaiser':
            window=windows.kaiser(nperseg,3.0)
        else:
            window=window_type
        freqs, times, Zxx = signal.stft(x_np, fs=self.fs, window=window,
                                        nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Zxx)
        mag_log = np.log1p(mag)
        stft_ = torch.tensor(mag_log, dtype=torch.float32)
        return self.min_max_scaler(stft_)

    def _resample_if_needed(self, data: np.ndarray, target_len: int) -> np.ndarray:
        from scipy.signal import resample
        if data.shape[1]!=target_len:
            return resample(data, target_len, axis=1)
        return data

    def cwt(self, x: torch.Tensor, wavelet_type='morlet', scales=None, target_length=None) -> torch.Tensor:
        try:
            import pywt
        except ImportError:
            raise ImportError("pywt requerido p/CWT.")
        x_np = x.cpu().numpy()
        n_samp=len(x_np)

        if wavelet_type.lower()=='morlet':
            wv='morl'
        elif wavelet_type.lower()=='ricker':
            wv='mexh'
        else:
            wv=wavelet_type

        if scales is None:
            n_scales=min(64,n_samp//2)
            scales_arr=np.logspace(0, np.log10(n_samp//2), n_scales)
        elif isinstance(scales,int):
            n_scales=min(scales,n_samp//2)
            scales_arr=np.logspace(0, np.log10(n_samp//2), n_scales)
        else:
            scales_arr=np.array(scales)

        import pywt
        coefs,_=pywt.cwt(x_np, scales_arr, wv, sampling_period=1/self.fs)
        mag = np.abs(coefs)

        if target_length:
            mag = self._resample_if_needed(mag, target_length)

        mag_log=np.log1p(mag)
        cwt_ = torch.tensor(mag_log, dtype=torch.float32)
        return self.min_max_scaler(cwt_)

    def dwt(self, x: torch.Tensor, wavelet='db4', level=None, target_length=None) -> torch.Tensor:
        try:
            import pywt
        except ImportError:
            raise ImportError("pywt requerido p/DWT.")
        x_np = x.cpu().numpy()
        wv=pywt.Wavelet(wavelet)
        n_samp=len(x_np)
        if level is None:
            lvl_auto=pywt.dwt_max_level(n_samp, wv.dec_len)
            level=min(lvl_auto,5)
        coeffs = pywt.wavedec(x_np, wavelet, level=level)
        details = coeffs[1:]
        if not details:
            return torch.zeros(1,1)

        if target_length is None:
            max_len = max(dc.shape[0] for dc in details)
            target_length=max_len

        stack_=[]

        for arr in details:
            arr_abs=np.log1p(np.abs(arr))
            arr_res=resample(arr_abs, target_length)
            stack_.append(arr_res)
        dwt_ = np.stack(stack_, axis=0)
        dwt_t = torch.tensor(dwt_, dtype=torch.float32)
        return self.min_max_scaler(dwt_t)
