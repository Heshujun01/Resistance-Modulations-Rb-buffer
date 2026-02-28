# Pythonscript for SUSTech Nanopore Analysis (PySNA)
#from abfheader import *
#from batchinfo import *
#from CUSUMV2 import detect_cusum
#from filterkit import *
#from peaktoolkit import *
#from PoreSizer import *
from scipy import ndimage
from scipy import signal
from scipy import io as spio
from scipy.interpolate import CubicSpline  # 3次样条插值CubicSpline
from scipy.optimize import curve_fit
import dbfload
import h5py
import hdf5plugin
#import loadmat
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pyabf
#import pyqtgraph as pg
#import pandas.io.parsers
import pandas as pd
import seaborn as sns
import sys

# ??????
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("analysis.log"), logging.StreamHandler()]
)

class PySNA:
    def __init__(self,baseline=200e-12,var=8e-12,threshold=180e-12):
        """
        ??? PySNA ??,???????
        """
        self.direc = os.getcwd()
        self.datafilename = []
        self.lr = []
        self.lastevent = []
        self.lastClicked = []
        self.hasbaselinebeenset = 0
        self.lastevent = 0
        self.deli = []
        self.imin = []
        self.frac = []
        self.dwell = []
        self.dt = []
        self.catdata = []
        self.colors = []
        self.sdf = pd.DataFrame(columns=[
            'fn', 'color', 'deli', 'frac', 'dwell', 'dt', 'noise', 'startpoints', 'endpoints'
        ])
        self.baseline = baseline
        self.var = var
        self.threshold = threshold

    def load(self, datafilename, lp_filter_cutoff=10000., output_samplerate=50000., channel=1):
        """
        ???????????
        
        ??:
            datafilename (str): ???????
            threshold (float): ??,??? 150 pA?
            lp_filter_cutoff (float): ?????????,??? 10 kHz?
            output_samplerate (float): ?????,??? 50 kHz?
            channel (int): ???,??? 1?
        """
        try:
            self.catdata = []
            self.batchinfo = pd.DataFrame(columns=['cutstart', 'cutend'])
            self.datafilename = datafilename
            self.lp_filter_cutoff = lp_filter_cutoff
            self.output_samplerate = output_samplerate

            # ??????????
            file_extension = os.path.splitext(datafilename)[1]
            if file_extension == '.log':
                self._load_log_file()
            elif file_extension == '.dat':
                self._load_dat_file()
            elif file_extension == '.txt':
                self._load_txt_file()
            elif file_extension == '.npy':
                self._load_npy_file()
            elif file_extension == '.abf':
                self._load_abf_file()
            elif file_extension == '.dbf':
                self._load_dbf_file()
            elif file_extension == '.fast5':
                self._load_fast5_file(channel)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # ???????????
            self._validate_samplerate_and_filter()

            # ?????
            self.t = np.arange(0, len(self.data)) / self.samplerate

            # ???????
            self.mean = np.median(self.data)
            self.std = np.std(self.data)
            if self.mean < 0.:
                self.data = -self.data
            logging.info(f'Baseline = {self.mean * 1e12:.2f} pA, RMS = {self.std * 1e12:.2f} pA')

        except Exception as e:
            logging.error(f"Error loading file {datafilename}: {e}")
            raise

    def _load_log_file(self):
        """?? .log ??"""
        try:
            matfilename = str(os.path.splitext(self.datafilename)[0])
            self.mat = spio.loadmat(matfilename + '_inf')
            matstruct = self.mat[os.path.basename(matfilename)]
            self.mat = matstruct[0][0]
            self.samplerate = float(self.mat['samplerate'])
            self.filtrate = float(self.mat['filterfreq'])
            logging.info(f"Successfully loaded .log file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .log file: {e}")
            raise
            
    def _load_dat_file(self):
        """?? .dat ??"""
        try:
            self.samplerate = 10000
            self.abflowpass = 10000
            self.data = np.fromfile(self.datafilename, dtype=np.int16)
            self.data = -(self.data - np.int16(-32768))/ 13.125 * 1e-12 # 反转数据方向
            self.matfilename = str(os.path.splitext(self.datafilename)[0])
            logging.info(f"Successfully loaded .dat file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .dat file: {e}")
            raise
            
    def _load_txt_file(self):
        """?? .txt ??"""
        try:
            self.data = pd.read_csv(self.datafilename, skiprows=1).values.flatten()
            self.matfilename = str(os.path.splitext(self.datafilename)[0])
            logging.info(f"Successfully loaded .txt file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .txt file: {e}")
            raise

    def _load_npy_file(self):
        """?? .npy ??"""
        try:
            self.data = np.load(self.datafilename)
            self.matfilename = str(os.path.splitext(self.datafilename)[0])
            logging.info(f"Successfully loaded .npy file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .npy file: {e}")
            raise

    def _load_abf_file(self):
        """?? .abf ??"""
        try:
            abf = pyabf.ABF(self.datafilename)
            self.data = abf.sweepY/1e12  # ??????????
            self.samplerate = abf.dataRate
            self.abflowpass = abf.dataRate
            logging.info(f"Successfully loaded .abf file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .abf file: {e}")
            raise
            
    def _load_dbf_file(self):
        """?? .dbf ??"""
        try:
            dbf, header = dbfload.dbfload(self.datafilename)
            dbf = np.array(dbf).T
            self.data = dbf[0]/32.767/1e12  # ??????????
            self.samplerate = 50000
            self.abflowpass = 50000
            logging.info(f"Successfully loaded .dbf file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .dbf file: {e}")
            raise
            
    def _load_fast5_file(self, channel):
        """?? .fast5 ??"""
        try:
            with h5py.File(self.datafilename, 'r') as f:
                raw_data = f[f'Raw/Channel_{channel}/Signal']
                self.data = raw_data[:] * 1e-12  # ???????
            self.samplerate = self.output_samplerate
            self.abflowpass = self.output_samplerate
            logging.info(f"Successfully loaded .fast5 file: {self.datafilename}")
        except Exception as e:
            logging.error(f"Error loading .fast5 file: {e}")
            raise

    def _validate_samplerate_and_filter(self):
        """
        ????????????
        """
        if self.output_samplerate > self.samplerate:
            logging.warning('Output samplerate cannot be higher than the original samplerate.')
            self.output_samplerate = self.samplerate

        if self.lp_filter_cutoff >= self.abflowpass:
            logging.warning('Already LP filtered lower than or at entry, data will not be filtered.')
            self.lp_filter_cutoff = self.abflowpass
        else:
            wn = round(self.lp_filter_cutoff / (self.samplerate / 2), 4)
            b, a = signal.bessel(4, wn, btype='low')
            self.data = signal.filtfilt(b, a, self.data)
            logging.info('Data filtered with LP filter cutoff: {} Hz'.format(self.lp_filter_cutoff))
        
    def plot_scat_hist(fig, handle_array, bins, color_list, x_range, y_range, x_label, y_label, legend_name, bounds_frac, bounds_rms, cmap='Greys', alpha=0.1, fitted=False, thr_x=None, thr_y=None):
        """
        ?????????,????????
        
        ??:
            fig: Matplotlib ?????
            handle_array (list): ????????
            bins (int): ????????
            color_list (list): ?????
            x_range (tuple): x ????
            y_range (tuple): y ????
            x_label (str): x ????
            y_label (str): y ????
            legend_name (str): ?????
            bounds_frac (tuple): ????????
            bounds_rms (tuple): RMS ??????
            cmap (str): ????,??? 'Greys'?
            alpha (float): ?????,??? 0.1?
            fitted (bool): ????????,??? False?
            thr_x (list): x ????,??? None?
            thr_y (list): y ????,??? None?
        """
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)

        plt.rc('font', family='Arial')  # ?? 'Helvetica'
        
        # ??:???
        for i, handle in enumerate(handle_array):
            sns.kdeplot(y=handle[1], x=handle[0], cmap=cmap, fill=True, bw_adjust=0.75)
            ax_main.plot(handle[0], handle[1], 'o', color=color_list[i], alpha=alpha, markersize=4)

        # ?????
        if thr_x:
            ax_main.plot(thr_x, y_range, 'k--')
        if thr_y:
            ax_main.plot(x_range, thr_y, 'k--')

        ax_main.set_xlim(x_range)
        ax_main.set_ylim(y_range)
        ax_main.set_xlabel(x_label, fontsize=32)
        ax_main.set_ylabel(y_label, fontsize=32)
        #ax_main.legend([legend_name], markerscale=0., fontsize=32, loc='best', frameon=False)

        # ?????
        for i, handle in enumerate(handle_array):
            ax_right.hist(handle[1], bins, range=y_range, weights=np.ones(len(handle[1])) / len(handle[1]),
                          facecolor=color_list[i], alpha=0.75, orientation='horizontal')

            if fitted:
                hist, bin_edges = np.histogram(handle[1], bins=np.arange(np.min(y_range), np.max(y_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_rms)
                x_fit = np.linspace(*y_range, 101)
                ax_right.plot(gaussian_mixture(x_fit, *popt), x_fit, color='red', label='fit')

        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_ylim(y_range)

        # ?????
        for i, handle in enumerate(handle_array):
            ax_top.hist(handle[0], bins, range=x_range, weights=np.ones(len(handle[0])) / len(handle[0]),
                        facecolor=color_list[i], alpha=0.75)

            if fitted:
                hist, bin_edges = np.histogram(handle[0], bins=np.arange(np.min(x_range), np.max(x_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_frac)
                x_fit = np.linspace(*x_range, 101)
                ax_top.plot(x_fit, gaussian_mixture(x_fit, *popt), color='red', label='fit')

        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_xlim(x_range)

    def plot_scat_hist(fig, handle_array, bins, color_list, x_range, y_range, x_label, y_label, legend_name, bounds_frac, bounds_rms, cmap='Greys', alpha=0.1, fitted=False, thr_x=None, thr_y=None):
        """
        ?????????,????????
        
        ??:
            fig: Matplotlib ?????
            handle_array (list): ????????
            bins (int): ????????
            color_list (list): ?????
            x_range (tuple): x ????
            y_range (tuple): y ????
            x_label (str): x ????
            y_label (str): y ????
            legend_name (str): ?????
            bounds_frac (tuple): ????????
            bounds_rms (tuple): RMS ??????
            cmap (str): ????,??? 'Greys'?
            alpha (float): ?????,??? 0.1?
            fitted (bool): ????????,??? False?
            thr_x (list): x ????,??? None?
            thr_y (list): y ????,??? None?
        """
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)

        plt.rc('font', family='Arial')  # ?? 'Helvetica'
        
        # ??:???
        for i, handle in enumerate(handle_array):
            sns.kdeplot(y=handle[1], x=handle[0], cmap=cmap, fill=True, bw_adjust=0.75)
            ax_main.plot(handle[0], handle[1], 'o', color=color_list[i], alpha=alpha, markersize=4)

        # ?????
        if thr_x:
            ax_main.plot(thr_x, y_range, 'k--')
        if thr_y:
            ax_main.plot(x_range, thr_y, 'k--')

        ax_main.set_xlim(x_range)
        ax_main.set_ylim(y_range)
        ax_main.set_xlabel(x_label, fontsize=32)
        ax_main.set_ylabel(y_label, fontsize=32)
        #ax_main.legend([legend_name], markerscale=0., fontsize=32, loc='best', frameon=False)

        # ?????
        for i, handle in enumerate(handle_array):
            ax_right.hist(handle[1], bins, range=y_range, weights=np.ones(len(handle[1])) / len(handle[1]),
                          facecolor=color_list[i], alpha=0.75, orientation='horizontal')

            if fitted:
                hist, bin_edges = np.histogram(handle[1], binsnp.arange(np.min(y_range), np.max(y_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_rms)
                x_fit = np.linspace(*y_range, 101)
                ax_right.plot(gaussian_mixture(x_fit, *popt), x_fit, color='red', label='fit')

        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_ylim(y_range)

        # ?????
        for i, handle in enumerate(handle_array):
            ax_top.hist(handle[0], bins, range=x_range, weights=np.ones(len(handle[0])) / len(handle[0]),
                        facecolor=color_list[i], alpha=0.75)

            if fitted:
                hist, bin_edges = np.histogram(handle[0], bins=np.arange(np.min(x_range), np.max(x_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_frac)
                x_fit = np.linspace(*x_range, 101)
                ax_top.plot(x_fit, gaussian_mixture(x_fit, *popt), color='red', label='fit')

        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_xlim(x_range)
        
    def scat_noise_frac(handle_array, bins=50, iave=[0, 1], irms=[0, 30], toff=[1e-1, 5e3], dtmin=2e-6, dtmax=1, ton=[1e-1, 1e3], data_loaded=False, color_list=[], title=[], bounds_frac=[], bounds_rms=[], cmap='Greys', alpha=0.1, fitted=False, thrx=[], thry=[]):
        """
        ????????????????
        
        ??:
            handle_array (list): ???? PySNA ??????
            bins (int): ???????,??? 50?
            iave (list): x ???,??? [0, 1]?
            irms (list): y ???,??? [0, 30]?
            toff (list): ??????,??? [0.1, 5000] ms?
            dtmin (float): ??????,??? 2 µs?
            dtmax (float): ??????,??? 1 s?
            ton (list): ??????,??? [0.1, 1000] ms?
            data_loaded (bool): ???????,??? False?
            color_list (list): ?????
            title (list): ???????
            bounds_frac (list): ????????
            bounds_rms (list): RMS ??????
            cmap (str): ????,??? 'Greys'?
            alpha (float): ?????,??? 0.1?
            fitted (bool): ????????,??? False?
            thrx (list): x ????,??? None?
            thry (list): y ????,??? None?
        """
        fig = plt.figure(figsize=(8, 8))
        
        plt.rc('font', family='Arial')  # ?? 'Helvetica'
        
        temp = [[handle.frac, handle.noise] for handle in handle_array]
        plot_scat_hist(fig, temp, bins, color_list, iave, irms, r'$\Delta$I/I$_0$', r'I$_{RMS}$ (pA)', title, bounds_frac, bounds_rms, cmap=cmap, alpha=alpha, fitted=fitted, thrx=thrx, thry=thry)

    def plot_scat_hist_base(fig, handle_array, bins, color_list, x_range, y_range, x_label, y_label, legend_name, bounds_x, bounds_y, log_scale=False, cmap='Greys', alpha=0.1, fitted=False, thrx=[], thry=[]):
        """
        ??????????????
        
        ??:
            fig: Matplotlib ?????
            handle_array (list): ????????
            bins (int): ????????
            color_list (list): ?????
            x_range (tuple): x ????
            y_range (tuple): y ????
            x_label (str): x ????
            y_label (str): y ????
            legend_name (str): ?????
            bounds_x (list): x ??????
            bounds_y (list): y ??????
            log_scale (bool): ????????,??? False?
            cmap (str): ????,??? 'Greys'?
            alpha (float): ?????,??? 0.1?
            fitted (bool): ????????,??? False?
            thrx (list): x ????,??? None?
            thry (list): y ????,??? None?
        """
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)

        plt.rc('font', family='Arial')  # ?? 'Helvetica'
        
        # ??:???
        for i, handle in enumerate(handle_array):
            if log_scale:
                sns.kdeplot(y=np.log10(handle[1]), x=handle[0], cmap=cmap, fill=True, bw_adjust=0.75)
                ax_main.plot(handle[0], np.log10(handle[1]), 'o', color=color_list[i], alpha=alpha, markersize=4)
            else:
                sns.kdeplot(y=handle[1], x=handle[0], cmap=cmap, fill=True, bw_adjust=0.75)
                ax_main.plot(handle[0], handle[1], 'o', color=color_list[i], alpha=alpha, markersize=4)

        # ?????
        if thrx:
            ax_main.plot(thrx, y_range, 'k--')
        if thry:
            ax_main.plot(x_range, thry, 'k--')

        ax_main.set_xlim(x_range)
        ax_main.set_ylim(y_range if not log_scale else np.log10(y_range))
        ax_main.set_xlabel(x_label, fontsize=32)
        ax_main.set_ylabel(y_label, fontsize=32)
        #ax_main.legend([legend_name], markerscale=0., fontsize=32, loc='best', frameon=False)

        # ?????
        for i, handle in enumerate(handle_array):
            if log_scale:
                ax_right.hist(np.log10(handle[1]), bins, range=np.log10(y_range),
                              weights=np.ones(len(handle[1])) / len(handle[1]),
                              facecolor=color_list[i], alpha=0.75, orientation='horizontal')
            else:
                ax_right.hist(handle[1], bins, range=y_range,
                              weights=np.ones(len(handle[1])) / len(handle[1]),
                              facecolor=color_list[i], alpha=0.75, orientation='horizontal')

            if fitted:
                hist, bin_edges = np.histogram(handle[1] if not log_scale else np.log10(handle[1]),
                                               bins=np.arange(np.min(y_range), np.max(y_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_y)
                x_fit = np.linspace(*y_range, 101) if not log_scale else np.linspace(*np.log10(y_range), 101)
                ax_right.plot(gaussian_mixture(x_fit, *popt), x_fit, color='red', label='fit')

        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_ylim(y_range if not log_scale else np.log10(y_range))

        # ?????
        for i, handle in enumerate(handle_array):
            ax_top.hist(handle[0], bins, range=x_range,
                        weights=np.ones(len(handle[0])) / len(handle[0]),
                        facecolor=color_list[i], alpha=0.75)

            if fitted:
                hist, bin_edges = np.histogram(handle[0], bins=np.arange(np.min(x_range), np.max(x_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_x)
                x_fit = np.linspace(*x_range, 101)
                ax_top.plot(x_fit, gaussian_mixture(x_fit, *popt), color='red', label='fit')

        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_xlim(x_range)

    def prepare_data_for_plot(handle_array, field_x, field_y, scale_factor=1.0):
        """
        ??????????
        
        ??:
            handle_array (list): ???? PySNA ??????
            field_x (str): x ?????
            field_y (str): y ?????
            scale_factor (float): ????,??? 1.0?
        
        ??:
            list: ??????????
        """
        return [[getattr(handle, field_x), getattr(handle, field_y) * scale_factor] for handle in handle_array]

    def prepare_data_for_analysis(handle_array, field_x, field_y, scale_factor=1.0):
        """
        ??????????
        
        ??:
            handle_array (list): ???? PySNA ??????
            field_x (str): x ?????
            field_y (str): y ?????
            scale_factor (float): ????,??? 1.0?
        
        ??:
            list: ??????????
        """
        return [[getattr(handle, field_x), getattr(handle, field_y) * scale_factor] for handle in handle_array]

    def plot_combined(fig, handle_array, bins, color_list, x_range, y_range, x_label, y_label, legend_name, bounds_x, bounds_y, log_scale=False, cmap='Greys', alpha=0.1, fitted=False, thrx=[], thry=[]):
        """
        ????????????
        
        ??:
            fig: Matplotlib ?????
            handle_array (list): ????????
            bins (int): ????????
            color_list (list): ?????
            x_range (tuple): x ????
            y_range (tuple): y ????
            x_label (str): x ????
            y_label (str): y ????
            legend_name (str): ?????
            bounds_x (list): x ??????
            bounds_y (list): y ??????
            log_scale (bool): ????????,??? False?
            cmap (str): ????,??? 'Greys'?
            alpha (float): ?????,??? 0.1?
            fitted (bool): ????????,??? False?
            thrx (list): x ????,??? None?
            thry (list): y ????,??? None?
        """
        gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
        ax_main = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)

        plt.rc('font', family='Arial')  # ?? 'Helvetica'
        
        # ??:???
        for i, handle in enumerate(handle_array):
            if log_scale:
                sns.kdeplot(y=np.log10(handle[1]), x=handle[0], cmap=cmap, fill=True, bw_adjust=0.75)
                ax_main.plot(handle[0], np.log10(handle[1]), 'o', color=color_list[i], alpha=alpha, markersize=4)
            else:
                sns.kdeplot(y=handle[1], x=handle[0], cmap=cmap, fill=True, bw_adjust=0.75)
                ax_main.plot(handle[0], handle[1], 'o', color=color_list[i], alpha=alpha, markersize=4)

        # ?????
        if thrx:
            ax_main.plot(thrx, y_range, 'k--')
        if thry:
            ax_main.plot(x_range, thry, 'k--')

        ax_main.set_xlim(x_range)
        ax_main.set_ylim(y_range if not log_scale else np.log10(y_range))
        ax_main.set_xlabel(x_label, fontsize=32)
        ax_main.set_ylabel(y_label, fontsize=32)
        #ax_main.legend([legend_name], markerscale=0., fontsize=32, loc='best', frameon=False)

        # ?????
        for i, handle in enumerate(handle_array):
            if log_scale:
                ax_right.hist(np.log10(handle[1]), bins, range=np.log10(y_range),
                              weights=np.ones(len(handle[1])) / len(handle[1]),
                              facecolor=color_list[i], alpha=0.75, orientation='horizontal')
            else:
                ax_right.hist(handle[1], bins, range=y_range,
                              weights=np.ones(len(handle[1])) / len(handle[1]),
                              facecolor=color_list[i], alpha=0.75, orientation='horizontal')

            if fitted:
                hist, bin_edges = np.histogram(handle[1] if not log_scale else np.log10(handle[1]),
                                               bins=np.arange(np.min(y_range), np.max(y_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_y)
                x_fit = np.linspace(*y_range, 101) if not log_scale else np.linspace(*np.log10(y_range), 101)
                ax_right.plot(gaussian_mixture(x_fit, *popt), x_fit, color='red', label='fit')

        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.set_ylim(y_range if not log_scale else np.log10(y_range))

        # ?????
        for i, handle in enumerate(handle_array):
            ax_top.hist(handle[0], bins, range=x_range,
                        weights=np.ones(len(handle[0])) / len(handle[0]),
                        facecolor=color_list[i], alpha=0.75)

            if fitted:
                hist, bin_edges = np.histogram(handle[0], bins=np.arange(np.min(x_range), np.max(x_range), 0.1), density=False)
                popt, _ = curve_fit(gaussian_mixture, bin_edges[:-1], hist, bounds=bounds_x)
                x_fit = np.linspace(*x_range, 101)
                ax_top.plot(x_fit, gaussian_mixture(x_fit, *popt), color='red', label='fit')

        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.set_xlim(x_range)
        
    def plot_together(handle_array, bins=50, iave=[0, 1], irms=[0, 30], toff=[1e-1, 5e3], dtmin=2e-6, dtmax=1, ton=[1e-1, 1e3], data_loaded=False, color_list=[], title=[], bounds_frac=[], bounds_rms=[], bounds_dwell=[], fitted=False, thrx=[], thry=[]):
        """
        ????????????
        
        ??:
            handle_array (list): ???? PySNA ??????
            bins (int): ???????,??? 50?
            iave (list): x ???,??? [0, 1]?
            irms (list): y ???,??? [0, 30]?
            toff (list): ??????,??? [0.1, 5000] ms?
            dtmin (float): ??????,??? 2 µs?
            dtmax (float): ??????,??? 1 s?
            ton (list): ??????,??? [0.1, 1000] ms?
            data_loaded (bool): ???????,??? False?
            color_list (list): ?????
            title (list): ???????
            bounds_frac (list): ????????
            bounds_rms (list): RMS ??????
            bounds_dwell (list): ??????????
            fitted (bool): ????????,??? False?
            thrx (list): x ????,??? None?
            thry (list): y ????,??? None?
        """
        fig = plt.figure(figsize=(24, 8))
        subfigs = fig.subfigures(1, 3)

        # ????:?? vs ??
        temp = prepare_data_for_analysis(handle_array, 'frac', 'noise')
        plot_combined(subfigs[0], temp, bins, color_list, iave, irms, r'$\Delta$I/I$_0$', r'I$_{RMS}$ (pA)', title,
                      bounds_frac, bounds_rms, fitted=fitted, thrx=thrx, thry=thry)

        # ????:???? vs ??
        temp = prepare_data_for_analysis(handle_array, 'frac', 'dwell', scale_factor=1e-3)
        plot_combined(subfigs[1], temp, bins, color_list, iave, toff, r'$\Delta$I/I$_0$', r'Dwell Time (ms)', title,
                      bounds_frac, bounds_dwell, log_scale=True, fitted=fitted, thrx=thrx, thry=thry)

        # ????:???? vs ??
        temp = prepare_data_for_analysis(handle_array, 'noise', 'dwell', scale_factor=1e-3)
        plot_combined(subfigs[2], temp, bins, color_list, irms, toff, r'I$_{RMS}$ (pA)', r'Dwell Time (ms)', title,
                      bounds_rms, bounds_dwell, log_scale=True, fitted=fitted, thrx=thrx, thry=thry)

        if data_loaded:
            # ???????,????????
            m, b = [], []
            for i, handle in enumerate(handle_array):
                hist, bin_edges = np.histogram(handle.dwell / 1e3, bins=bins * 250, density=True)
                x = bin_edges[:15] + (bin_edges[1] - bin_edges[0]) / 2
                y = np.log10(hist[:15] / hist[0])
                x = np.delete(x, np.isinf(y))
                y = np.delete(y, np.isinf(y))
                slope, intercept = np.polyfit(x, y, 1)
                m.append(slope)
                b.append(intercept)
                print(f"Fitting Params #{i}: Slope={slope:.4f}, Intercept={intercept:.4f}")

            # ?????????
            plt.figure(figsize=(20, 4))
            plt.subplot(1, 3, 1)
            for i, handle in enumerate(handle_array):
                plt.hist(np.log10(handle.dwell[1:] / 1e3), bins, density=True, facecolor=color_list[i], alpha=0.75)
            plt.xlim(toff)
            plt.xlabel(r'log t$_{off}$ (ms)')

            plt.subplot(1, 3, 2)
            for i, handle in enumerate(handle_array):
                hist, bin_edges = np.histogram(handle.dwell / 1e3, bins=bins * 250, density=True)
                plt.plot(bin_edges[1:], 10**(m[i] * bin_edges[1:] + b[i]) * hist[0], color_list[i])
            plt.xlim([0, 10**toff[1] / 4])
            plt.xlabel(r't$_{off}$ (ms)')
            plt.title(r't$_{off}$ = ' + f"{-1 / m[i] * 1000:.1f} ms")

            plt.subplot(1, 3, 3)
            for i, handle in enumerate(handle_array):
                hist, bin_edges = np.histogram(handle.dwell / 1e3, bins=bins * 250, density=True)
                plt.semilogy(bin_edges[1:], hist / hist[0], color_list[i], 'o')
                plt.semilogy(bin_edges[1:], 10**(m[i] * bin_edges[:-1] + b[i]), color_list[i])
            plt.xlim([0, 10**toff[1] / 4])
            plt.xlabel(r't$_{off}$ (ms)')
            plt.ylim([2e-3, 5])
            plt.title(r't$_{off}$ = ' + f"{-1 / m[i] * 1000:.1f} ms")

    def plot_temporal_and_hist(self, baseline=None, var=None, threshold=None, xlim=None, ylim=[-25., 375.], analyzed=False, hist_scale=None, color='g', channel=1):
        """
        绘制时间序列数据及其直方图。
        
        参数：
            baseline (float): 基线值，默认为 None。
            var (float): 数据的标准差，默认为 None。
            threshold (float): 阈值，默认为 None。
            xlim (list): 时间轴范围，默认为 None。
            ylim (list): 电流范围，默认为 [-25, 375]。
            analyzed (bool): 是否绘制分析结果，默认为 False。
            hist_scale (float): 直方图的 x 轴范围，默认为 None。
            color (str): 绘图颜色，默认为 'k'（黑色）。
            channel (int): 通道编号，默认为 1。
        """
        # 创建图形和网格布局
        fig = plt.figure(figsize=(20, 8))
        grid = plt.GridSpec(8, 8, hspace=0.25, wspace=0.25)
        
        # 设置字体大小
        plt.rc('font', family='Arial', size=32)
        plt.rc('axes', labelsize=40)
        
        # 主图：时间序列
        ax_main = fig.add_subplot(grid[1:-1, 0:-1])
        ax_main.plot(self.t[2:][:-2], self.data[2:][:-2] * 10**12, color='k', label="Current")
        
        # 绘制基线和阈值线
        if baseline is not None and var is not None:
            ax_main.plot([self.t[2], self.t[-2]], [baseline * 10**12, baseline * 10**12], '--', color='b', label="Baseline")
            ax_main.plot([self.t[2], self.t[-2]], [(baseline + 3 * var) * 10**12] * 2, '--', color='b', label="+3σ")
            ax_main.plot([self.t[2], self.t[-2]], [(baseline - 3 * var) * 10**12] * 2, '--', color='b', label="-3σ")
        else:
            ax_main.plot([self.t[2], self.t[-2]], [self.baseline * 10**12] * 2, '--', color='b', label="Baseline")
            ax_main.plot([self.t[2], self.t[-2]], [(self.baseline + 3 * self.var) * 10**12] * 2, '--', color='b', label="+3σ")
            ax_main.plot([self.t[2], self.t[-2]], [(self.baseline - 3 * self.var) * 10**12] * 2, '--', color='b', label="-3σ")
        
        if threshold is not None:
            ax_main.plot([self.t[2], self.t[-2]], [threshold * 10**12] * 2, '--', color='r', label="Threshold")
        else:
            ax_main.plot([self.t[2], self.t[-2]], [self.threshold * 10**12] * 2, '--', color='r', label="Threshold")
        
        # 如果已分析，绘制事件标记
        if analyzed:
            for i in range(len(self.startpoints)):
                ax_main.plot(
                    self.t[self.startpoints[i] - 5:self.endpoints[i] + 5],
                    self.data[self.startpoints[i] - 5:self.endpoints[i] + 5] * 10**12,
                    color=color
                )
        
        # 设置主图的坐标轴范围和标签
        if xlim is not None:
            ax_main.set_xlim(xlim)
        else:
            ax_main.margins(0., 0.)
        ax_main.set_ylim(ylim)
        ax_main.set_xlabel('Time (s)')
        ax_main.set_ylabel('Current (pA)')
        #ax_main.legend(loc='best')
        
        # 右侧直方图
        if hist_scale is not None:
            ax_right = fig.add_subplot(grid[1:-1, -1], yticklabels=[])
            n, bins, patches = ax_right.hist(
                self.data * 10**12, bins=np.arange(np.min(ylim), np.max(ylim), 0.1), density=True, facecolor=color, alpha=0.75, orientation='horizontal'
            )
            
            if baseline is not None and var is not None:
                ax_right.plot([0, hist_scale], [baseline * 10**12] * 2, '--', color='b')
                ax_right.plot([0, hist_scale], [(baseline + 3 * var) * 10**12] * 2, '--', color='b')
                ax_right.plot([0, hist_scale], [(baseline - 3 * var) * 10**12] * 2, '--', color='b')
            else:
                ax_right.plot([0, hist_scale], [self.baseline * 10**12] * 2, '--', color='b')
                ax_right.plot([0, hist_scale], [(self.baseline + 3 * self.var) * 10**12] * 2, '--', color='b')
                ax_right.plot([0, hist_scale], [(self.baseline - 3 * self.var) * 10**12] * 2, '--', color='b')
            
            if threshold is not None:
                ax_right.plot([0, hist_scale], [threshold * 10**12] * 2, '--', color='r')
            else:
                ax_right.plot([0, hist_scale], [self.threshold * 10**12] * 2, '--', color='r')
            
            ax_right.set_xlim([0, hist_scale])
            ax_right.set_ylim(ylim)
        
        # 显示图形
        plt.show()
    
        
    def crop_trace(self, t_begin=None, t_end=None):
        if t_begin!=None or t_end!=None:
            flag = np.zeros(len(self.data), dtype=bool)
            if t_begin == None:
                # 删除开始的部分
                flag[:int(t_end*self.output_samplerate)] = 1
            elif t_end == None:
                # 删除结束的部分
                flag[int(t_begin*self.output_samplerate):] = 1
            else:
                # 删除中间的部分
                flag[int(t_begin*self.output_samplerate):int(t_end*self.output_samplerate)] = 1
        
            self.data = np.delete(self.data,flag)
            self.t = np.arange(0,len(self.data))
            self.t = self.t/self.output_samplerate
            
    def remove_gating_events(self, baseline=None, var=None, check_length=50, delete_length=1250, var_fold=4):
        """
        Remove gating events from the data based on specified conditions.
    
        Parameters:
            baseline (float): Baseline current value. If None, use self.baseline.
            var (float): Variance of the current. If None, use self.var.
            check_length (int): Number of points to check for gating events after an event ends.
            delete_length (int): Number of points to delete if a gating event is detected.
            var_fold (float): Multiplier for variance to define the gating threshold.
        """
        # Update baseline and variance if provided
        if baseline is not None:
            self.baseline = baseline
        if var is not None:
            self.var = var
    
        # Create a flag array to mark points to be removed
        flag = np.zeros(len(self.data), dtype=bool)
    
        # Iterate through each event to detect gating events
        for i, dwell in enumerate(self.dwell):
            # Check if there is a gating event after the event ends
            if np.any(self.data[int(self.endpoints[i]):int(self.endpoints[i] + check_length)] > self.baseline + var_fold * self.var):
                # Mark the entire event and surrounding region for deletion
                flag[int(self.startpoints[i]):int(self.endpoints[i] + delete_length)] = True
    
            # Additional condition: Check if the event contains negative values
            if np.any(self.data[int(self.startpoints[i]):int(self.endpoints[i])] < 0):
                flag[int(self.startpoints[i]):int(self.endpoints[i])] = True
    
        # Remove flagged points from the data
        self.data = np.delete(self.data, flag)
        self.t = np.arange(0, len(self.data)) / self.output_samplerate
    
        # Recalculate startpoints and endpoints after removing flagged points
        cumulative_flag = np.cumsum(flag)  # Cumulative sum of flags to adjust indices
        self.startpoints = self.startpoints - np.array([cumulative_flag[int(sp)] for sp in self.startpoints])
        self.endpoints = self.endpoints - np.array([cumulative_flag[int(ep)] for ep in self.endpoints])
    
        # Filter out invalid events
        valid_indices = (self.startpoints >= 0) & (self.endpoints < len(self.data))
        self.startpoints = self.startpoints[valid_indices]
        self.endpoints = self.endpoints[valid_indices]
        self.numberofevents = len(self.startpoints)
    
        logging.info(f'Remaining events: {self.numberofevents:.0f}')   
        
    def statistical_plots(self, bins=50, frac_range=[0,1], dwell_range=[1e-1, 5e3], noise_range=[0,35], data_loaded=False):
        """
        Generate statistical plots for the current analysis results.
    
        Parameters:
            bins (int): Number of bins for histograms.
            Imin (float): Minimum current deviation for plotting.
            Imax (float): Maximum current deviation for plotting.
            dtmin (float): Minimum inter-event interval for plotting.
            dtmax (float): Maximum inter-event interval for plotting.
            ton (list): Range for log(t_on) plot.
            toff (list): Range for log(t_off) plot.
            data_loaded (bool): Whether the data has been preloaded for additional analysis.
        """
        # Update the statistics DataFrame
        self.sdf = pd.DataFrame(columns=['deli', 'imin', 'frac', 'dwell', 'dt', 'noise', 'startpoints', 'endpoints'])
        self.sdf = pd.concat([
            self.sdf,
            pd.DataFrame({
                'deli': self.deli,
                'imin': self.imin,
                'frac': self.frac,
                'dwell': self.dwell,
                'dt': self.dt,
                'noise': self.noise,
                'startpoints': self.startpoints,
                'endpoints': self.endpoints
            })
        ], ignore_index=True)
    
        # Create a figure with subplots
        fig = plt.figure(figsize=(28, 12))
        subfigs = fig.subfigures(1, 3)
    
        # Plot 1: Fraction vs Noise
        self._plot_fraction_vs_noise(subfigs[0], bins, frac_range, dwell_range, noise_range, data_loaded)
    
        # Plot 2: Fraction vs Dwell Time
        self._plot_fraction_vs_dwell(subfigs[1], bins, frac_range, dwell_range, noise_range, data_loaded)
    
        # Plot 3: Noise vs Dwell Time
        self._plot_noise_vs_dwell(subfigs[2], bins, frac_range, dwell_range, noise_range, data_loaded)
    
        # Additional analysis if data is loaded
        if data_loaded:
            self._fit_and_plot_dwell_time(bins, toff)
            self._fit_and_plot_inter_event_interval(bins, ton)
    
        # Save results
        #self.save()
        #self.savetarget()
    
    def _plot_fraction_vs_noise(self, fig, bins, frac_range, dwell_range, noise_range, data_loaded):
        """
        Plot fraction vs noise with marginal histograms.
        """
        grid = plt.GridSpec(7, 7, hspace=0.25, wspace=0.25)
        plt.rc('font', family='Arial', size=32)
        plt.rc('axes', labelsize=40)
    
        # Main scatter plot
        ax_main = fig.add_subplot(grid[1:-1, 0:-1])
        ax_main.plot(self.frac, self.noise, 'o', color='#1f77b4', alpha=0.05)
        ax_main.margins(0., 0.)
        ax_main.set_xlim(frac_range)
        ax_main.set_ylim(noise_range)
        ax_main.set_xlabel(r'$\Delta$I/I$_0$')
        ax_main.set_ylabel(r'I$_{RMS}$ (pA)')
    
        # Right histogram
        ax_right = fig.add_subplot(grid[1:-1, -1], yticklabels=[])
        ax_right.hist(self.noise, bins, density=True, color='#1f77b4', alpha=0.75, orientation='horizontal')
        ax_right.margins(0., 0.)
        ax_right.set_ylim(noise_range)
    
        # Top histogram
        ax_top = fig.add_subplot(grid[0, :-1], xticklabels=[])
        if data_loaded:
            temp = self.data[self.data < self.threshold]
            ax_top.hist(1 - temp / self.baseline, bins, density=True, facecolor='k', alpha=0.75)
        ax_top.hist(self.frac, bins, density=True, color='#1f77b4', alpha=0.75)
        ax_top.margins(0., 0.)
        ax_top.set_xlim(frac_range)
    
    def _plot_fraction_vs_dwell(self, fig, bins, frac_range, dwell_range, noise_range, data_loaded):
        """
        Plot fraction vs dwell time with marginal histograms.
        """
        grid = plt.GridSpec(7, 7, hspace=0.25, wspace=0.25)
        plt.rc('font', family='Arial', size=32)
        plt.rc('axes', labelsize=40)
    
        # Main scatter plot
        ax_main = fig.add_subplot(grid[1:-1, 0:-1])
        ax_main.semilogy(self.frac, self.dwell / 1e3, 'o', color='#ff7f0e', alpha=0.05)
        ax_main.set_xlim(frac_range)
        ax_main.set_ylim(dwell_range)
        ax_main.set_xlabel('$\Delta$I/$I_0$')
        ax_main.set_ylabel('Dwell Time (ms)')
    
        # Right histogram
        ax_right = fig.add_subplot(grid[1:-1, -1], yticklabels=[])
        ax_right.hist(np.log10(self.dwell / 1e3), bins, density=True, color='#ff7f0e', alpha=0.75, orientation='horizontal')
        ax_right.margins(0., 0.)
        ax_right.set_ylim(np.log10(dwell_range))
    
        # Top histogram
        ax_top = fig.add_subplot(grid[0, :-1], xticklabels=[])
        if data_loaded:
            temp = self.data[self.data < self.threshold]
            ax_top.hist(1 - temp / self.baseline, bins, density=True, facecolor='k', alpha=0.75)
        ax_top.hist(self.frac, bins, density=True, color='#ff7f0e', alpha=0.75)
        ax_top.margins(0., 0.)
        ax_top.set_xlim(frac_range)
    
    def _plot_noise_vs_dwell(self, fig, bins, frac_range, dwell_range, noise_range, data_loaded):
        """
        Plot noise vs dwell time with marginal histograms.
        """
        grid = plt.GridSpec(7, 7, hspace=0.25, wspace=0.25)
        plt.rc('font', family='Arial', size=32)
        plt.rc('axes', labelsize=40)
    
        # Main scatter plot
        ax_main = fig.add_subplot(grid[1:-1, 0:-1])
        ax_main.semilogy(self.noise, self.dwell / 1e3, 'o', color='#2ca02c', alpha=0.05)
        ax_main.margins(0., 0.)
        ax_main.set_xlim(noise_range)
        ax_main.set_ylim(dwell_range)
        ax_main.set_xlabel('I$_{RMS}$ (pA)')
        ax_main.set_ylabel('Dwell Time (ms)')
    
        # Right histogram
        ax_right = fig.add_subplot(grid[1:-1, -1], yticklabels=[])
        ax_right.hist(np.log10(self.dwell / 1e3), bins, density=True, color='#2ca02c', alpha=0.75, orientation='horizontal')
        ax_right.margins(0., 0.)
        ax_right.set_ylim(np.log10(dwell_range))
    
        # Top histogram
        ax_top = fig.add_subplot(grid[0, :-1], xticklabels=[])
        ax_top.hist(self.noise, bins, density=True, color='#2ca02c', alpha=0.75)
        ax_top.margins(0., 0.)
        ax_top.set_xlim(noise_range)
    
    def _fit_and_plot_dwell_time(self, bins, toff):
        """
        Fit and plot dwell time distributions.
        """
        hist, bin_edges = np.histogram(self.dwell / 1e3, bins=bins, density=True)
        x = bin_edges[:15] + (bin_edges[1] - bin_edges[0]) / 2
        y = np.log10(hist[:15] / hist[0])
        x = np.delete(x, np.isinf(y))
        y = np.delete(y, np.isinf(y))
        m, b = np.polyfit(x, y, 1)
        print("Fitting Params (Dwell Time):", round(m, 4), round(b, 4))
    
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 3, 1)
        plt.hist(np.log10(self.dwell[1:] / 1e3), bins, density=True, facecolor='k', alpha=0.75)
        plt.xlim(toff)
        plt.xlabel(r'log $t_{off}$ (ms)')
    
        plt.subplot(1, 3, 2)
        plt.hist(self.dwell / 1e3, bins, density=True, facecolor='k', alpha=0.75)
        plt.plot(bin_edges[1:], 10**(m * bin_edges[1:] + b) * hist[0], 'r')
        plt.xlim([0, 10**toff[1] / 4])
        plt.xlabel('$t_{off}$ (ms)')
        plt.title('$t_{off}$ = ' + str(round(-1 / m, 1)) + ' ms')
    
        plt.subplot(1, 3, 3)
        plt.semilogy(bin_edges[1:], hist / hist[0], 'ko')
        plt.semilogy(bin_edges[1:], 10**(m * bin_edges[:-1] + b), 'k')
        plt.xlim([0, 10**toff[1] / 4])
        plt.xlabel('$t_{off}$ (ms)')
        plt.ylim([2e-3, 5])
        plt.title('$t_{off}$ = ' + str(round(-1 / m, 1)) + ' ms')
    
    def _fit_and_plot_inter_event_interval(self, bins, ton):
        """
        Fit and plot inter-event interval distributions.
        """
        hist, bin_edges = np.histogram(self.dt * 1000, bins=bins, density=True)
        x = bin_edges[:15] + (bin_edges[1] - bin_edges[0]) / 2
        y = np.log10(hist[:15] / hist[0])
        x = np.delete(x, np.isinf(y))
        y = np.delete(y, np.isinf(y))
        m, b = np.polyfit(x, y, 1)
        print("Fitting Params (Inter-event Interval):", round(m, 4), round(b, 4))
    
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 3, 1)
        plt.hist(np.log10(self.dt[1:] * 1000), bins, density=True, facecolor='k', alpha=0.75)
        plt.xlim(ton)
        plt.xlabel(r'log $t_{on}$ (ms)')
    
        plt.subplot(1, 3, 2)
        plt.hist(self.dt * 1000, bins, density=True, facecolor='k', alpha=0.75)
        plt.plot(bin_edges[1:], 10**(m * bin_edges[1:] + b) * hist[0], 'r')
        plt.xlim([0, 10**ton[1] / 4])
        plt.xlabel('$t_{on}$ (ms)')
        plt.title('$t_{on}$ = ' + str(round(-1 / m * 1000, 1)) + ' ms')
    
        plt.subplot(1, 3, 3)
        plt.semilogy(bin_edges[1:], hist / hist[0], 'ko')
        plt.semilogy(bin_edges[1:], 10**(m * bin_edges[:-1] + b), 'k')
        plt.xlim([0, 10**ton[1] / 4])
        plt.xlabel('$t_{on}$ (ms)')
        plt.ylim([5e-3, 2])
        plt.title('$t_{on}$ = ' + str(round(-1 / m * 1000, 1)) + ' ms')
        
    def fine_analyze(self, baseline=None, var=None, threshold=None, residue=5, median=False):
        """
        Perform fine-grained analysis of events in the data.
    
        Parameters:
            baseline (float): Baseline current value.
            var (float): Variance of the current.
            threshold (float): Threshold for event detection.
            residue (int): Adjustment factor for event boundaries.
        """
        # Update parameters if provided
        if baseline is not None:
            self.baseline = baseline
        if var is not None:
            self.var = var
        if threshold is not None:
            self.threshold = threshold
    
        # Step 1: Find all points below the threshold
        below = np.where(self.data < self.threshold)[0]
    
        # Step 2: Locate event start and end points
        startpoints, endpoints = self._find_event_boundaries(below)
    
        # Step 3: Refine event boundaries and calculate properties
        self._refine_and_calculate_properties(startpoints, endpoints, residue, median)
    
        # Step 4: Print summary statistics
        self._print_summary_statistics()
    
    def _find_event_boundaries(self, below):
        """Find start and end points of events."""
        startandend = np.diff(below)
        startpoints = np.insert(startandend, 0, 2)
        endpoints = np.insert(startandend, -1, 2)
        startpoints = np.where(startpoints > 1)[0]
        endpoints = np.where(endpoints > 1)[0]
        startpoints = below[startpoints]
        endpoints = below[endpoints]
    
        # Remove events at the edges of the file
        if startpoints[0] == 0:
            startpoints = np.delete(startpoints, 0)
            endpoints = np.delete(endpoints, 0)
        if endpoints[-1] == len(self.data) - 1:
            startpoints = np.delete(startpoints, -1)
            endpoints = np.delete(endpoints, -1)
    
        return startpoints, endpoints
    
    def _refine_and_calculate_properties(self, startpoints, endpoints, residue, median = False):
        """Refine event boundaries and calculate event properties."""
        highthresh = self.threshold
        numberofevents = len(startpoints)
    
        # Track back to baseline to refine start and end points
        for j in range(numberofevents):
            sp, ep = self._track_to_baseline(startpoints[j], endpoints[j], highthresh)
            startpoints[j], endpoints[j] = sp, ep
    
        # Filter out invalid events
        valid_indices = (startpoints != 0) & (endpoints != 0)
        startpoints = startpoints[valid_indices]
        endpoints = endpoints[valid_indices]
        
        # Adjust boundaries based on local minima
        self.deli, self.dwell, self.imin = [], [], []
        for i in range(len(startpoints)):
            self.dwell.append((endpoints[i] - startpoints[i]) * 1e6 / self.output_samplerate)
            sp, ep, deli, imin = self._adjust_boundaries(startpoints[i], endpoints[i], residue, median)
            startpoints[i], endpoints[i] = sp, ep
            self.deli.append(deli)
            self.imin.append(imin)
            #self.dwell.append((ep - sp) * 1e6 / self.output_samplerate)

        # Update event properties
        self.startpoints = startpoints
        self.endpoints = endpoints
        self.deli = np.array(self.deli)
        self.imin = np.array(self.imin)
        self.dwell = np.array(self.dwell)
        self.frac = self.deli / self.baseline
        self.dt = np.append(0, np.diff(startpoints) / self.output_samplerate)
        self.noise = (10**12) * np.array([np.std(self.data[x+2:endpoints[i]-2]) for i, x in enumerate(startpoints)])
        self.numberofevents = len(self.startpoints)
    
    def _track_to_baseline(self, sp, ep, highthresh):
        """Track back to baseline to refine start and end points."""
        while self.data[sp] < highthresh and sp > 0:
            sp -= 1
    
        if ep == len(self.data)-1:  # sure that the current returns to baseline
            ep = 0              # before file ends. If not, mark points for
            sp = 0              # deletion and break from loop

        while self.data[ep] < highthresh and ep > 0 and ep < len(self.data) - 1:
            ep += 1
            if ep == len(self.data) -1:  # sure that the current returns to baseline
                    ep = 0              # before file ends. If not, mark points for
                    sp = 0              # deletion and break from loop
                        
        return sp, ep
    
    def _adjust_boundaries(self, sp, ep, residue, median=False):
        """Adjust event boundaries based on local minima."""
        mins = signal.argrelmin(self.data[sp:ep])[0] + sp
        mins = mins[self.data[mins] < self.baseline - 5 * self.var]
    
        if len(mins) == 1:
            sp, ep = self._adjust_single_minima(sp, ep, mins[0], residue)
        elif len(mins) > 1:
            sp, ep = self._adjust_multiple_minima(sp, ep, mins, residue)
    
        if median:
            deli = self.baseline - np.median(self.data[sp:ep-2]) #maybe use local_baseline later on
        else:
            deli = self.baseline - np.mean(self.data[sp:ep-2]) #maybe use local_baseline later on
            
        imin = self.baseline - np.min(self.data[sp:ep]) 
        return sp, ep, deli, imin
    
    def _adjust_single_minima(self, sp, ep, min_point, residue):
        """Adjust boundaries for a single minimum."""
        sp = self._find_adjusted_startpoint(min_point, residue)
        ep = self._find_adjusted_endpoint(min_point, residue)
        return sp, ep
    
    def _adjust_multiple_minima(self, sp, ep, mins, residue):
        """Adjust boundaries for multiple minima."""
        sp = self._find_adjusted_startpoint(mins[0], residue)
        ep = self._find_adjusted_endpoint(mins[-1], residue)
        return sp, ep
    
    def _find_adjusted_startpoint(self, min_point, residue):
        """Find the adjusted start point."""
        for offset in [residue * 4, residue * 2, residue * 1.5, residue, residue * 0.5, residue * 0.25]:
            if self.data[min_point - int(offset)] < self.threshold - 3 * self.var:
                return min_point - int(offset)
        return min_point
    
    def _find_adjusted_endpoint(self, min_point, residue):
        """Find the adjusted end point."""
        for offset in [residue * 4, residue * 2, residue * 1.5, residue, residue * 0.5, residue * 0.25]:
            if self.data[min_point + int(offset)] < self.threshold:
                return min_point + int(offset)
        return min_point + 2
    
    def _print_summary_statistics(self):
        """Print summary statistics of the analysis."""
        logging.info(f'Total Events: {self.numberofevents:.0f}')
        logging.info(f'Current Deviation: {np.mean(self.deli * 10**12):.2f} pA')
        logging.info(f'Dwell Time: {np.median(self.dwell):.2f} μs')
        logging.info(f'Event Rate: {self.numberofevents / self.t[-1]:.1f} events/s')

    def regular_analyze_origin(self, baseline=None, var=None, threshold=None, residue=5):
        global startpoints,endpoints, mins
        self.analyzetype = 'fine'
    
        if baseline!=None:
            self.baseline = baseline
        if var!=None:
            self.var = var
        if threshold!=None:
            self.threshold = threshold
    
    #### find all points below threshold ####
    
        below = np.where(self.data < self.threshold)[0]
    
    #### locate the points where the current crosses the threshold ####
    
        startandend = np.diff(below)
        startpoints = np.insert(startandend, 0, 2)
        endpoints = np.insert(startandend, -1, 2)
        startpoints = np.where(startpoints>1)[0]
        endpoints = np.where(endpoints>1)[0]
        startpoints = below[startpoints]
        endpoints = below[endpoints]
    
    #### Eliminate events that start before file or end after file ####
    
        if startpoints[0] == 0:
            startpoints = np.delete(startpoints,0)
            endpoints = np.delete(endpoints,0)
        if endpoints [-1] == len(self.data)-1:
            startpoints = np.delete(startpoints,-1)
            endpoints = np.delete(endpoints,-1)
    
    #### Track points back up to baseline to find true start and end ####
    
        numberofevents=len(startpoints)
        highthresh = self.baseline - self.var
    
        for j in range(numberofevents):
            sp = startpoints[j] #mark initial guess for starting point
            while self.data[sp] < highthresh and sp > 0:
                sp = sp-1 # track back until we return to baseline
            startpoints[j] = sp # mark true startpoint
    
            ep = endpoints[j] #repeat process for end point
            if ep == len(self.data) -1:  # sure that the current returns to baseline
                endpoints[j] = 0              # before file ends. If not, mark points for
                startpoints[j] = 0              # deletion and break from loop
                ep = 0
                break
            while self.data[ep] < highthresh:
                ep = ep+1
                if ep == len(self.data) -1:  # sure that the current returns to baseline
                    endpoints[j] = 0              # before file ends. If not, mark points for
                    startpoints[j] = 0              # deletion and break from loop
                    ep = 0
                    break
                else:
                    try:
                        if ep > startpoints[j+1]: # if we hit the next startpoint before we
                            startpoints[j+1] = 0    # return to baseline, mark for deletion
                            endpoints[j] = 0                  # and break out of loop
                            ep = 0
                            break
                    except:
                        IndexError
                endpoints[j] = ep
    
        startpoints = startpoints[startpoints!=0] # delete those events marked for
        endpoints = endpoints[endpoints!=0]       # deletion earlier
        self.numberofevents = len(startpoints)
    
        if len(startpoints) > len(endpoints):
            startpoints = np.delete(startpoints, -1)
            self.numberofevents = len(startpoints)
    
    
    #### Now we want to move the endpoints to be the last minimum for each ####
    #### event so we find all minimas for each event, and set endpoint to last ####
    
        self.deli = np.zeros(self.numberofevents)
        self.imin = np.zeros(self.numberofevents)
        self.dwell = np.zeros(self.numberofevents)
    
        for i in range(self.numberofevents):
            self.dwell[i] = (endpoints[i]-startpoints[i])*1e6/self.output_samplerate  
            
            mins = np.array(signal.argrelmin(self.data[startpoints[i]:endpoints[i]])[0] + startpoints[i])
            mins = mins[self.data[mins] < self.baseline - 5*self.var]
            if len(mins) == 1:
                #pass
                if self.data[mins[0]-residue*4]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue*4
                elif self.data[mins[0]-residue*2]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue*2
                elif self.data[mins[0]-int(residue*1.5)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue*1.5)
                elif self.data[mins[0]-residue]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue
                elif self.data[mins[0]-int(residue/2.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/2.)
                elif self.data[mins[0]-int(residue/4.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/4.)
                else:
                    startpoints[i] = mins[0]
                if self.data[mins[0]+residue*4]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+residue*4
                elif self.data[mins[0]+residue*2]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+residue*2
                elif self.data[mins[0]+int(residue*1.5)]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+int(residue*1.5)
                elif self.data[mins[0]+residue]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+residue
                elif self.data[mins[0]+int(residue/2.)]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+int(residue/2.)
                elif self.data[mins[0]+int(residue/4.)]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+int(residue/4.)
                else:
                    endpoints[i] = mins[0]+2
                self.deli[i] = self.baseline - np.mean(self.data[startpoints[i]:endpoints[i]]) 
                self.imin[i] = self.baseline - np.min(self.data[startpoints[i]:endpoints[i]])
            elif len(mins) > 1:
                #startpoints[i] = mins[0]-residue
                #endpoints[i] = mins[-1]+residue
                if self.data[mins[0]-residue]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue
                elif self.data[mins[0]-int(residue/2.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/2.)
                elif self.data[mins[0]-int(residue/4.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/4.)
                else:
                    startpoints[i] = mins[0]
                if self.data[mins[-1]+residue]<self.threshold:
                    endpoints[i] = mins[-1]+residue
                elif self.data[mins[-1]+int(residue/2.)]<self.threshold:
                    endpoints[i] = mins[-1]+int(residue/2.)
                elif self.data[mins[-1]+int(residue/4.)]<self.threshold:
                    endpoints[i] = mins[-1]+int(residue/4.)
                else:
                    endpoints[i] = mins[-1]+2
                self.deli[i] = self.baseline - np.mean(self.data[startpoints[i]:endpoints[i]])
                self.imin[i] = self.baseline - np.min(self.data[startpoints[i]:endpoints[i]])
    
        print(startpoints)
        print(endpoints)
        startpoints = startpoints[self.deli!=0]
        self.startpoints = startpoints
        endpoints = endpoints[self.deli!=0]
        self.endpoints = endpoints
        self.dwell = self.dwell[self.deli!=0]
        self.imin = self.imin[self.deli!=0]
        self.deli = self.deli[self.deli!=0]
        self.frac = self.deli/self.baseline
        self.dt=np.append(0,np.diff(startpoints)/self.output_samplerate)
        self.numberofevents = len(self.dt)
        self.noise = (10**12)*np.array([np.std(self.data[x:endpoints[i]]) for i,x in enumerate(startpoints)])
    
        print('Total Events: '+str(self.numberofevents))
        print('Current Deviation: '+str(round(np.mean(self.deli*10**12),2))+' pA')
        print('Dwell Time: '+str(round(np.median(self.dwell),2))+ u' μs')
        print('Event Rate: '+str(round(self.numberofevents/self.t[-1],1))+' events/s')

    def parametric_free_cusum_analysis(self, notshow=False, noFrame=False, eventbuffer=500, color='k', dashline=0.5, baseline=None):
        """
        Perform parametric-free CUSUM analysis on events.
        """
        # Initialize variables
        eventtime = [0]
        frac = self.frac
        deli = self.deli
        imin = self.imin
        startpoints = self.startpoints
        endpoints = self.endpoints
    
        # Dynamically estimate parameters
        mindwell = 1.0 / self.lp_filter_cutoff  # Minimum dwell time based on LP filter cutoff
        minfrac = 5.0 * self.var / self.baseline  # Minimum fractional current deviation
    
        for i, dwell in enumerate(self.dwell):
            if i < len(self.dwell) - 1 and dwell > mindwell and frac[i] > minfrac:
                # Check for overlapping events
                if endpoints[i] + eventbuffer > startpoints[i + 1]:
                    if not noFrame:
                        print(f"#{i}: Overlapping event")
                    frac[i] = np.nan
                    deli[i] = np.nan
                    continue
    
                # Extract event data with buffer
                eventdata = self.data[int(startpoints[i] - eventbuffer):int(endpoints[i] + eventbuffer)]
                eventtime = np.arange(0, len(eventdata)) + eventbuffer + eventtime[-1]
    
                # Estimate baseline dynamically
                if baseline is None:
                    temp_baseline = np.median([
                        self.data[int(startpoints[i] - eventbuffer):int(startpoints[i])],
                        self.data[int(endpoints[i]):int(endpoints[i] + eventbuffer)]
                    ])
                else:
                    temp_baseline = baseline
    
                # Adaptive CUSUM detection
                basesd = np.std(eventdata[:eventbuffer])  # Baseline noise estimate
                dt = 1 / self.output_samplerate  # Sampling interval
    
                # Use adaptive thresholds and step sizes
                cusumthresh = 5.0  # Initial threshold (can be adjusted adaptively)
                cusumstep = 3.0  # Initial step size (can be adjusted adaptively)
    
                # Perform CUSUM analysis
                cusum = detect_cusum(
                    data=eventdata,
                    basesd=basesd,
                    dt=dt,
                    threshold=cusumthresh,
                    stepsize=cusumstep,
                    minlength=mindwell * self.output_samplerate,
                    maxstates=25
                )
    
                # Iteratively adjust thresholds if fewer than 3 levels are detected
                while len(cusum['CurrentLevels']) < 3:
                    cusumthresh *= 0.9
                    cusumstep *= 0.9
                    cusum = detect_cusum(
                        data=eventdata,
                        basesd=basesd,
                        dt=dt,
                        threshold=cusumthresh,
                        stepsize=cusumstep,
                        minlength=mindwell * self.output_samplerate,
                        maxstates=25
                    )
                    if not noFrame:
                        print(f"#{i}: Not Sensitive Enough")
    
                # Update event properties based on CUSUM results
                frac[i] = (np.max(cusum['CurrentLevels']) - np.min(cusum['CurrentLevels'])) / np.max(cusum['CurrentLevels'])
                deli[i] = np.max(cusum['CurrentLevels']) - np.min(cusum['CurrentLevels'])
    
                # Optional: Plot results
                if not notshow:
                    plt.figure(figsize=(1, 2))
                    if noFrame:
                        ax = plt.axes()
                        ax.spines['left'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                    plt.plot(eventtime / self.output_samplerate, eventdata / temp_baseline, color=color)
                    plt.plot(eventtime / self.output_samplerate, dashline * np.ones(np.size(eventtime)), color='gray', linestyle='dashed')
                    plt.plot(eventtime / self.output_samplerate, 1.0 * np.ones(np.size(eventtime)), color='gray', linestyle='dashed')
                    plt.plot(eventtime / self.output_samplerate, 0.0 * np.ones(np.size(eventtime)), color='gray', linestyle='dashed')
                    if not noFrame:
                        plt.plot(eventtime[eventbuffer] / self.output_samplerate, self.data[int(startpoints[i])] / temp_baseline, 'o', color='g')
                        plt.plot(eventtime[-eventbuffer] / self.output_samplerate, self.data[int(endpoints[i])] / temp_baseline, 'o', color='r')
                    plt.ylim([-0.25, 1.25])
                    if not noFrame:
                        plt.title(f"#{i}: {round(self.noise[i], 1)}")
                    plt.savefig(f"{i}.svg", dpi=300)
    
        # Clean up invalid events
        valid_indices = np.isfinite(deli)
        self.dwell = self.dwell[valid_indices]
        self.dt = self.dt[valid_indices]
        self.noise = self.noise[valid_indices]
        self.frac = frac[valid_indices]
        self.deli = deli[valid_indices]
        self.imin = imin[valid_indices]
        self.startpoints = self.startpoints[valid_indices]
        self.endpoints = self.endpoints[valid_indices]
    
        # Save results
        #np.savetxt(
        #    self.matfilename + '_cusum_DB.txt',
        #    np.column_stack((self.deli, self.frac, self.dwell, self.dt, self.noise)),
        #    delimiter='\t',
        #    header="deli\tfrac\tdwell\tdt\tnoise"
        #)
        #Functions        

    def regular_analyze_median(self, baseline=None, var=None, threshold=None, residue=5):
        global startpoints,endpoints, mins
        self.analyzetype = 'fine'
    
        if baseline!=None:
            self.baseline = baseline
        if var!=None:
            self.var = var
        if threshold!=None:
            self.threshold = threshold
    
    #### find all points below threshold ####
    
        below = np.where(self.data < self.threshold)[0]
    
    #### locate the points where the current crosses the threshold ####
    
        startandend = np.diff(below)
        startpoints = np.insert(startandend, 0, 2)
        endpoints = np.insert(startandend, -1, 2)
        startpoints = np.where(startpoints>1)[0]
        endpoints = np.where(endpoints>1)[0]
        startpoints = below[startpoints]
        endpoints = below[endpoints]
    
    #### Eliminate events that start before file or end after file ####
    
        if startpoints[0] == 0:
            startpoints = np.delete(startpoints,0)
            endpoints = np.delete(endpoints,0)
        if endpoints [-1] == len(self.data)-1:
            startpoints = np.delete(startpoints,-1)
            endpoints = np.delete(endpoints,-1)
    
    #### Track points back up to baseline to find true start and end ####
    
        numberofevents=len(startpoints)
        highthresh = self.baseline - self.var
    
        for j in range(numberofevents):
            sp = startpoints[j] #mark initial guess for starting point
            while self.data[sp] < highthresh and sp > 0:
                sp = sp-1 # track back until we return to baseline
            startpoints[j] = sp # mark true startpoint
    
            ep = endpoints[j] #repeat process for end point
            if ep == len(self.data) -1:  # sure that the current returns to baseline
                endpoints[j] = 0              # before file ends. If not, mark points for
                startpoints[j] = 0              # deletion and break from loop
                ep = 0
                break
            while self.data[ep] < highthresh:
                ep = ep+1
                if ep == len(self.data) -1:  # sure that the current returns to baseline
                    endpoints[j] = 0              # before file ends. If not, mark points for
                    startpoints[j] = 0              # deletion and break from loop
                    ep = 0
                    break
                else:
                    try:
                        if ep > startpoints[j+1]: # if we hit the next startpoint before we
                            startpoints[j+1] = 0    # return to baseline, mark for deletion
                            endpoints[j] = 0                  # and break out of loop
                            ep = 0
                            break
                    except:
                        IndexError
                endpoints[j] = ep
    
        startpoints = startpoints[startpoints!=0] # delete those events marked for
        endpoints = endpoints[endpoints!=0]       # deletion earlier
        self.numberofevents = len(startpoints)
    
        if len(startpoints) > len(endpoints):
            startpoints = np.delete(startpoints, -1)
            self.numberofevents = len(startpoints)
    
    
    #### Now we want to move the endpoints to be the last minimum for each ####
    #### event so we find all minimas for each event, and set endpoint to last ####
    
        self.deli = np.zeros(self.numberofevents)
        self.imin = np.zeros(self.numberofevents)
        self.dwell = np.zeros(self.numberofevents)
    
        for i in range(self.numberofevents):
            self.dwell[i] = (endpoints[i]-startpoints[i])*1e6/self.output_samplerate  
            
            mins = np.array(signal.argrelmin(self.data[startpoints[i]:endpoints[i]])[0] + startpoints[i])
            mins = mins[self.data[mins] < self.baseline - 5*self.var]
            if len(mins) == 1:
                #pass
                if self.data[mins[0]-residue*4]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue*4
                elif self.data[mins[0]-residue*2]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue*2
                elif self.data[mins[0]-int(residue*1.5)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue*1.5)
                elif self.data[mins[0]-residue]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue
                elif self.data[mins[0]-int(residue/2.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/2.)
                elif self.data[mins[0]-int(residue/4.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/4.)
                else:
                    startpoints[i] = mins[0]
                if self.data[mins[0]+residue*4]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+residue*4
                elif self.data[mins[0]+residue*2]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+residue*2
                elif self.data[mins[0]+int(residue*1.5)]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+int(residue*1.5)
                elif self.data[mins[0]+residue]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+residue
                elif self.data[mins[0]+int(residue/2.)]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+int(residue/2.)
                elif self.data[mins[0]+int(residue/4.)]<self.threshold-3*self.var:
                    endpoints[i] = mins[0]+int(residue/4.)
                else:
                    endpoints[i] = mins[0]+2
                self.deli[i] = self.baseline - np.mean(self.data[startpoints[i]:endpoints[i]]) 
                self.imin[i] = self.baseline - np.min(self.data[startpoints[i]:endpoints[i]]) 
                
            elif len(mins) > 1:
                #startpoints[i] = mins[0]-residue
                #endpoints[i] = mins[-1]+residue
                if self.data[mins[0]-residue]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-residue
                elif self.data[mins[0]-int(residue/2.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/2.)
                elif self.data[mins[0]-int(residue/4.)]<self.threshold-3*self.var:
                    startpoints[i] = mins[0]-int(residue/4.)
                else:
                    startpoints[i] = mins[0]
                if self.data[mins[-1]+residue]<self.threshold:
                    endpoints[i] = mins[-1]+residue
                elif self.data[mins[-1]+int(residue/2.)]<self.threshold:
                    endpoints[i] = mins[-1]+int(residue/2.)
                elif self.data[mins[-1]+int(residue/4.)]<self.threshold:
                    endpoints[i] = mins[-1]+int(residue/4.)
                else:
                    endpoints[i] = mins[-1]+2
                self.deli[i] = self.baseline - np.median(self.data[startpoints[i]:endpoints[i]])
                elf.deli[i] = self.baseline - np.min(self.data[startpoints[i]:endpoints[i]])
    
        print(startpoints)
        print(endpoints)
        startpoints = startpoints[self.deli!=0]
        self.startpoints = startpoints
        endpoints = endpoints[self.deli!=0]
        self.endpoints = endpoints
        self.dwell = self.dwell[self.deli!=0]
        self.imin = self.imin[self.deli!=0]
        self.deli = self.deli[self.deli!=0]
        self.frac = self.deli/self.baseline
        self.dt=np.append(0,np.diff(startpoints)/self.output_samplerate)
        self.numberofevents = len(self.dt)
        self.noise = (10**12)*np.array([np.std(self.data[x+2:endpoints[i]-2]) for i,x in enumerate(startpoints)])
    
        print('Total Events: '+str(self.numberofevents))
        print('Current Deviation: '+str(round(np.mean(self.deli*10**12),2))+' pA')
        print('Dwell Time: '+str(round(np.median(self.dwell),2))+ u' μs')
        print('Event Rate: '+str(round(self.numberofevents/self.t[-1],1))+' events/s')
            
    def baseline_fit( self, kernel_size=101, time_range=(0, 300), current_range=(200e-12, 250e-12) ):
        """
        中值滤波+筛选区间+线性基线拟合
        返回筛选后的时间、电流和拟合参数 slope, intercept
        """
        # 1. 中值滤波
        current_filtered = signal.medfilt(self.data, kernel_size=kernel_size)
        
        # 2. 按时间和电流区间筛选
        mask = (
            (self.t >= time_range[0]) & (self.t <= time_range[1]) &
            (current_filtered > current_range[0]) & (current_filtered < current_range[1])
        )
        test_time = self.t[mask]
        test_current = current_filtered[mask]
        
        # 3. 线性拟合
        slope, intercept = np.polyfit(test_time, test_current, 1)
        
        return test_time, test_current, slope, intercept

    def crop_dwell_snaHandle(snaHandle, t_min=1e3, t_max=1e6):
        """
        ?????????????
        
        ??:
            snaHandle: PySNA ???
            t_min (float): ??????,??? 1 ms?
            t_max (float): ??????,??? 1 s?
        
        ??:
            ???? PySNA ???
        """
        condition = (snaHandle.dwell >= t_min) & (snaHandle.dwell <= t_max)
        snaHandle.deli = snaHandle.deli[condition]
        snaHandle.imin = snaHandle.imin[condition]
        snaHandle.frac = snaHandle.frac[condition]
        snaHandle.noise = snaHandle.noise[condition]
        snaHandle.dt = snaHandle.dt[condition]
        snaHandle.startpoints = snaHandle.startpoints[condition]
        snaHandle.endpoints = snaHandle.endpoints[condition]
        
        snaHandle.dwell = snaHandle.dwell[condition]
        snaHandle.numberofevents = len(snaHandle.startpoints)
        return snaHandle

    def crop_noise_snaHandle(snaHandle, rms_min=0, rms_max=100):
        """
        ?????????????
        
        ??:
            snaHandle: PySNA ???
            t_min (float): ??????,??? 1 ms?
            t_max (float): ??????,??? 1 s?
        
        ??:
            ???? PySNA ???
        """
        condition = (snaHandle.noise >= rms_min) & (snaHandle.noise <= rms_max)
        snaHandle.deli = snaHandle.deli[condition]
        snaHandle.imin = snaHandle.imin[condition]
        snaHandle.frac = snaHandle.frac[condition]
        snaHandle.dwell = snaHandle.dwell[condition]
        snaHandle.dt = snaHandle.dt[condition]
        snaHandle.startpoints = snaHandle.startpoints[condition]
        snaHandle.endpoints = snaHandle.endpoints[condition]
        
        snaHandle.noise = snaHandle.noise[condition]
        snaHandle.numberofevents = len(snaHandle.startpoints)
        return snaHandle

    def crop_frac_snaHandle(snaHandle, I_min=0., I_max=1.):
        """
        ?????????????
        
        ??:
            snaHandle: PySNA ???
            t_min (float): ??????,??? 1 ms?
            t_max (float): ??????,??? 1 s?
        
        ??:
            ???? PySNA ???
        """
        condition = (snaHandle.frac >= I_min) & (snaHandle.frac <= I_max)
        snaHandle.deli = snaHandle.deli[condition]
        snaHandle.imin = snaHandle.imin[condition]
        snaHandle.noise = snaHandle.noise[condition]
        snaHandle.dwell = snaHandle.dwell[condition]
        snaHandle.dt = snaHandle.dt[condition]
        snaHandle.startpoints = snaHandle.startpoints[condition]
        snaHandle.endpoints = snaHandle.endpoints[condition]
                
        snaHandle.frac = snaHandle.frac[condition]
        snaHandle.numberofevents = len(snaHandle.startpoints)
        return snaHandle
        
    # 定义通用的多高斯函数
    def multi_gaussian(x, *params):
        """
        多高斯函数模型
        :param x: 自变量
        :param params: 参数列表，每三个参数代表一个高斯峰 (幅度、中心位置、标准差)
        :return: 多高斯函数的值
        """
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            a, x0, sigma = params[i:i + 3]
            y += a * np.exp(-(x - x0)**2 / (2 * sigma**2))
        return y
    
    # 动态生成初始参数猜测
    def generate_initial_guess(num_peaks = 1, peak_centers = [0.5], peak_amp = [1.], peak_sigma = [0.01]):
        """
        根据指定的高斯峰数量生成初始参数猜测
        :param num_peaks: 高斯峰数量
        :param peak_centers: 直方图的 bin 中心位置
        :param peak_amp: 直方图的计数值
        :return: 初始参数列表
        """
        initial_guess = []
        peak_indices = np.linspace(0, len(peak_centers) - 1, num_peaks + 2)[1:-1].astype(int)
        for idx in peak_indices:
            amplitude = peak_amp[idx]
            center = peak_centers[idx]
            sigma = peak_sigma[idx]  # 默认标准差
            initial_guess.extend([amplitude, center, sigma])
        return initial_guess