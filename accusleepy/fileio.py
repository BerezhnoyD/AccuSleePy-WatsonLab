import json
import os
import h5py
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QListWidget, QListWidgetItem, QPushButton, 
                             QLineEdit, QMessageBox, QGroupBox, QAbstractItemView)
from PySide6.QtCore import Qt

from accusleepy.brain_state_set import BRAIN_STATES_KEY, BrainState, BrainStateSet
import accusleepy.constants as c



@dataclass
class EMGFilter:
    """Convenience class for a EMG filter parameters"""

    order: int  # filter order
    bp_lower: int | float  # lower bandpass frequency
    bp_upper: int | float  # upper bandpass frequency


@dataclass
class Hyperparameters:
    """Convenience class for model training hyperparameters"""

    batch_size: int
    learning_rate: float
    momentum: float
    training_epochs: int

    
class ChannelSelectorDialog(QDialog):
    """Class for handling the opening of the nwb file"""


    def __init__(self, nwb_path, total_duration, fs, n_channels, channel_names=None):
        super().__init__()
        self.setWindowTitle("Select Data to Load")
        self.resize(500, 600)
        
        self.result_data = None
        self.nwb_path = nwb_path
        self.fs = fs
        self.total_duration = total_duration
        
        # Main Layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # --- Info Section ---
        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f"File: {os.path.basename(nwb_path)}"))
        info_layout.addWidget(QLabel(f"Duration: {total_duration:.2f} sec"))
        info_layout.addWidget(QLabel(f"Rate: {fs} Hz"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # --- Time Selection ---
        time_group = QGroupBox("Time Range (Seconds)")
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Start:"))
        self.txt_start = QLineEdit("0")
        time_layout.addWidget(self.txt_start)
        time_layout.addWidget(QLabel("Duration:"))
        self.txt_duration = QLineEdit("10")
        time_layout.addWidget(self.txt_duration)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)
        
        # --- Channel Selection ---
        lists_layout = QHBoxLayout()
        
        # EEG List (Single Selection)
        eeg_group = QGroupBox("Select ONE EEG Channel")
        eeg_layout = QVBoxLayout()
        self.list_eeg = QListWidget()
        self.list_eeg.setSelectionMode(QAbstractItemView.SingleSelection)
        eeg_layout.addWidget(self.list_eeg)
        eeg_group.setLayout(eeg_layout)
        lists_layout.addWidget(eeg_group)

        # EMG List (Single Selection)
        emg_group = QGroupBox("Select ONE EMG Channel")
        emg_layout = QVBoxLayout()
        self.list_emg = QListWidget()
        self.list_emg.setSelectionMode(QAbstractItemView.SingleSelection)
        emg_layout.addWidget(self.list_emg)
        emg_group.setLayout(emg_layout)
        lists_layout.addWidget(emg_group)
        
        layout.addLayout(lists_layout)

        # Populate both lists
        for i in range(n_channels):
            name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"
            
            # Add to EEG
            item_eeg = QListWidgetItem(name)
            item_eeg.setData(Qt.UserRole, i)
            self.list_eeg.addItem(item_eeg)
            
            # Add to EMG
            item_emg = QListWidgetItem(name)
            item_emg.setData(Qt.UserRole, i)
            self.list_emg.addItem(item_emg)
            
        # Default Selection
        if n_channels > 0:
            self.list_eeg.item(0).setSelected(True)
            if n_channels > 1:
                self.list_emg.item(1).setSelected(True)
            else:
                self.list_emg.item(0).setSelected(True)
        
        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self.load_data)
        btn_load.setStyleSheet("font-weight: bold; padding: 5px;")
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def load_data(self):
        try:
            # 1. Validate Time
            start = float(self.txt_start.text())
            dur = float(self.txt_duration.text())
            
            if start < 0 or start >= self.total_duration:
                raise ValueError("Invalid Start Time")
            if dur <= 0:
                raise ValueError("Duration must be positive")
            if start + dur > self.total_duration:
                QMessageBox.warning(self, "Warning", "Truncating duration to file end.")
                dur = self.total_duration - start

            # 2. Get Single Selections
            eeg_items = self.list_eeg.selectedItems()
            emg_items = self.list_emg.selectedItems()
            
            if not eeg_items:
                raise ValueError("Please select one EEG channel.")
            if not emg_items:
                raise ValueError("Please select one EMG channel.")
            
            eeg_idx = eeg_items[0].data(Qt.UserRole)
            eeg_name = eeg_items[0].text()
            
            emg_idx = emg_items[0].data(Qt.UserRole)
            emg_name = emg_items[0].text()
            
            # Prepare indices and names list
            # Order: EEG first, then EMG
            indices = [eeg_idx, emg_idx]
            names = [f"eeg", f"emg"]
            
            # 3. Load Data
            print(f"Loading indices {indices} from {start}s to {start+dur}s...")
            data_matrix = self._read_nwb_chunk(indices, start, dur)
            
            # 4. Create DataFrame
            time_vector = np.linspace(start, start+dur, data_matrix.shape[0], endpoint=False)
            
            df = pd.DataFrame(data_matrix, columns=names)
            
            self.result_data = df
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _read_nwb_chunk(self, indices, start, duration):
        with h5py.File(self.nwb_path, 'r') as f:
            dset = None
            if 'acquisition' in f:
                acq = f['acquisition']
                if 'ElectricalSeries' in acq:
                    dset = acq['ElectricalSeries']['data']
                elif len(acq.keys()) > 0:
                    dset = acq[list(acq.keys())[0]]['data']
            
            if dset is None:
                raise KeyError("Data not found.")

            start_samp = int(start * self.fs)
            end_samp = start_samp + int(duration * self.fs)
            
            # Optimization: Load only necessary columns
            # h5py requires sorted indices for efficient I/O
            sorted_idx = sorted(list(set(indices))) # unique & sorted
            data_chunk = dset[start_samp:end_samp, sorted_idx]
            
            # Reorder columns to match [EEG, EMG] request
            # Create a map: where did my requested index end up in the sorted list?
            # requested: [5, 2] -> sorted: [2, 5] -> data columns: [0 (was 2), 1 (was 5)]
            # we need output to be [col 1, col 0]
            
            # Map index value -> column position in loaded chunk
            val_to_col = {val: i for i, val in enumerate(sorted_idx)}
            
            # Build reordered array
            final_data = np.zeros((data_chunk.shape[0], len(indices)))
            for i, target_idx in enumerate(indices):
                col_in_chunk = val_to_col[target_idx]
                final_data[:, i] = data_chunk[:, col_in_chunk]
                
            return final_data
    


@dataclass
class Recording:
    """Store information about a recording"""

    name: int = 1  # name to show in the GUI
    recording_file: str = ""  # path to recording file
    label_file: str = ""  # path to label file
    calibration_file: str = ""  # path to calibration file
    sampling_rate: int | float = 0.0  # sampling rate, in Hz
    widget: QListWidgetItem = None  # list item widget shown in the GUI


def load_calibration_file(filename: str) -> (np.array, np.array):
    """Load a calibration file

    :param filename: filename
    :return: mixture means and SDs
    """
    df = pd.read_csv(filename)
    mixture_means = df[c.MIXTURE_MEAN_COL].values
    mixture_sds = df[c.MIXTURE_SD_COL].values
    return mixture_means, mixture_sds

def load_nwb(filename:str) -> pd.DataFrame:
    """Load a nwb file as a dataframe
    Raise a popup window suggesting to choose channels
    
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None

    try:
        with h5py.File(filename, 'r') as f:
            dset = None
            series = None
            if 'acquisition' in f:
                acq = f['acquisition']
                if 'ElectricalSeries' in acq:
                    series = acq['ElectricalSeries']
                elif len(acq.keys()) > 0:
                    series = acq[list(acq.keys())[0]]
            
            if series is None:
                raise ValueError("Invalid NWB.")
            
            dset = series['data']
            shape = dset.shape
            
            fs = 250.0
            if 'starting_time' in series:
                fs = series['starting_time'].attrs['rate']
            elif 'rate' in series:
                fs = series['rate'][()]
                
            duration = shape[0] / fs
            n_channels = shape[1]
            names = [f"Ch {i}" for i in range(n_channels)]

    except Exception as e:
        print(f"Failed to read metadata: {e}")
        return None

    dialog = ChannelSelectorDialog(filename, duration, fs, n_channels, names)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.result_data
    else:
        return None
    
       
    

def load_csv_or_parquet(filename: str) -> pd.DataFrame:
    """Load a csv or parquet file as a dataframe

    :param filename: filename
    :return: dataframe of file contents
    """
    extension = os.path.splitext(filename)[1]
    if extension == ".csv":
        df = pd.read_csv(filename)
    elif extension == ".parquet":
        df = pd.read_parquet(filename)
    elif extension == ".nwb":
        df = load_nwb(filename)
    else:
        raise Exception("file must be csv or parquet or nwb")
    return df


def load_recording(filename: str) -> (np.array, np.array):
    """Load recording of EEG and EMG time series data

    :param filename: filename
    :return: arrays of EEG and EMG data
    """
    
    df = load_csv_or_parquet(filename)
    eeg = df[c.EEG_COL].values
    emg = df[c.EMG_COL].values
    
    if c.SL_WAVE_COL in df.columns:
        sl_wave = df[c.SL_WAVE_COL].values
    else:
        sl_wave = None
    
    return eeg, emg, sl_wave


def load_labels(filename: str) -> (np.array, np.array):
    """Load file of brain state labels and confidence scores

    :param filename: filename
    :return: array of brain state labels and, optionally, array of confidence scores
    """
    df = load_csv_or_parquet(filename)
    if c.CONFIDENCE_SCORE_COL in df.columns:
        return df[c.BRAIN_STATE_COL].values, df[c.CONFIDENCE_SCORE_COL].values
    else:
        return df[c.BRAIN_STATE_COL].values, None


def save_labels(
    labels: np.array, filename: str, 
    spectr_data: np.array = None, spectr_freq: np.array = None, confidence_scores: np.array = None,
) -> None:
    """Save brain state labels to file

    :param labels: brain state labels
    :param filename: filename
    :param confidence_scores: optional confidence scores
    """
    if confidence_scores is not None:
        pd.DataFrame(
            {c.BRAIN_STATE_COL: labels, c.CONFIDENCE_SCORE_COL: confidence_scores}
        ).to_csv(filename, index=False)
    else:
        pd.DataFrame({c.BRAIN_STATE_COL: labels}).to_csv(filename, index=False)
        
    np.save(filename[:-3] + '_spectr.npy', spectr_data)
    np.save(filename[:-3] + '_spectr_freq.npy', spectr_freq)
    


def load_config() -> tuple[
    BrainStateSet,
    int | float,
    bool,
    bool,
    int | float,
    EMGFilter,
    Hyperparameters,
    int,
    bool,
]:
    """Load configuration file with brain state options

    :return: set of brain state options,
        default epoch length,
        default overwrite setting,
        default confidence score output setting,
        default minimum bout length,
        EMG filter parameters,
        model training hyperparameters,
        default epochs to show for manual scoring,
        default autoscroll state for manual scoring
    """
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), c.CONFIG_FILE), "r"
    ) as f:
        data = json.load(f)

    return (
        BrainStateSet(
            [BrainState(**b) for b in data[BRAIN_STATES_KEY]], c.UNDEFINED_LABEL
        ),
        data[c.DEFAULT_EPOCH_LENGTH_KEY],
        data.get(c.DEFAULT_OVERWRITE_KEY, c.DEFAULT_OVERWRITE_SETTING),
        data.get(c.DEFAULT_CONFIDENCE_SETTING_KEY, c.DEFAULT_CONFIDENCE_SETTING),
        data.get(c.DEFAULT_MIN_BOUT_LENGTH_KEY, c.DEFAULT_MIN_BOUT_LENGTH),
        EMGFilter(
            **data.get(
                c.EMG_FILTER_KEY,
                {
                    "order": c.DEFAULT_EMG_FILTER_ORDER,
                    "bp_lower": c.DEFAULT_EMG_BP_LOWER,
                    "bp_upper": c.DEFAULT_EMG_BP_UPPER,
                },
            )
        ),
        Hyperparameters(
            **data.get(
                c.HYPERPARAMETERS_KEY,
                {
                    "batch_size": c.DEFAULT_BATCH_SIZE,
                    "learning_rate": c.DEFAULT_LEARNING_RATE,
                    "momentum": c.DEFAULT_MOMENTUM,
                    "training_epochs": c.DEFAULT_TRAINING_EPOCHS,
                },
            )
        ),
        data.get(c.EPOCHS_TO_SHOW_KEY, c.DEFAULT_EPOCHS_TO_SHOW),
        data.get(c.AUTOSCROLL_KEY, c.DEFAULT_AUTOSCROLL_STATE),
    )


def save_config(
    brain_state_set: BrainStateSet,
    default_epoch_length: int | float,
    overwrite_setting: bool,
    save_confidence_setting: bool,
    min_bout_length: int | float,
    emg_filter: EMGFilter,
    hyperparameters: Hyperparameters,
    epochs_to_show: int,
    autoscroll_state: bool,
) -> None:
    """Save configuration of brain state options to json file

    :param brain_state_set: set of brain state options
    :param default_epoch_length: default epoch length
    :param save_confidence_setting: default setting for
        saving confidence scores
    :param emg_filter: EMG filter parameters
    :param min_bout_length: default minimum bout length
    :param overwrite_setting: default setting for overwriting
        existing labels
    :param hyperparameters: model training hyperparameters
    :param epochs_to_show: default epochs to show for manual scoring,
    :param autoscroll_state: default autoscroll state for manual scoring
    """
    output_dict = brain_state_set.to_output_dict()
    output_dict.update({c.DEFAULT_EPOCH_LENGTH_KEY: default_epoch_length})
    output_dict.update({c.DEFAULT_OVERWRITE_KEY: overwrite_setting})
    output_dict.update({c.DEFAULT_CONFIDENCE_SETTING_KEY: save_confidence_setting})
    output_dict.update({c.DEFAULT_MIN_BOUT_LENGTH_KEY: min_bout_length})
    output_dict.update({c.EMG_FILTER_KEY: emg_filter.__dict__})
    output_dict.update({c.HYPERPARAMETERS_KEY: hyperparameters.__dict__})
    output_dict.update({c.EPOCHS_TO_SHOW_KEY: epochs_to_show})
    output_dict.update({c.AUTOSCROLL_KEY: autoscroll_state})
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), c.CONFIG_FILE), "w"
    ) as f:
        json.dump(output_dict, f, indent=4)


def load_recording_list(filename: str) -> list[Recording]:
    """Load list of recordings from file

    :param filename: filename of list of recordings
    :return: list of recordings
    """
    with open(filename, "r") as f:
        data = json.load(f)
    recording_list = [Recording(**r) for r in data[c.RECORDING_LIST_NAME]]
    for i, r in enumerate(recording_list):
        r.name = i + 1
    return recording_list


def save_recording_list(filename: str, recordings: list[Recording]) -> None:
    """Save list of recordings to file

    :param filename: where to save the list
    :param recordings: list of recordings to export
    """
    recording_dict = {
        c.RECORDING_LIST_NAME: [
            {
                "recording_file": r.recording_file,
                "label_file": r.label_file,
                "calibration_file": r.calibration_file,
                "sampling_rate": r.sampling_rate,
            }
            for r in recordings
        ]
    }
    with open(filename, "w") as f:
        json.dump(recording_dict, f, indent=4)
