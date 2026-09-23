import sys
import os
import h5py
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QListWidget, QListWidgetItem, QPushButton, 
                             QLineEdit, QMessageBox, QGroupBox, QAbstractItemView)
from PyQt5.QtCore import Qt

class ChannelSelectorDialog(QDialog):
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
            names = [f"EEG: {eeg_name}", f"EMG: {emg_name}"]
            
            # 3. Load Data
            print(f"Loading indices {indices} from {start}s to {start+dur}s...")
            data_matrix = self._read_nwb_chunk(indices, start, dur)
            
            # 4. Create DataFrame
            time_vector = np.linspace(start, start+dur, data_matrix.shape[0], endpoint=False)
            
            df = pd.DataFrame(data_matrix, columns=names)
            df.insert(0, "Time", time_vector)
            
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

def load_nwb_with_gui(nwb_path):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    if not os.path.exists(nwb_path):
        print(f"Error: {nwb_path} not found.")
        return None

    try:
        with h5py.File(nwb_path, 'r') as f:
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
            
            fs = 1250.0
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

    dialog = ChannelSelectorDialog(nwb_path, duration, fs, n_channels, names)
    if dialog.exec_() == QDialog.Accepted:
        return dialog.result_data
    else:
        return None

if __name__ == "__main__":
    test_file = "F:/NFSS_cohort0/12_15_25-LD0/12_16_25-LD1/12_16_25-LD1.nwb"
    df = load_nwb_with_gui(test_file)
    if df is not None:
        print("\nLoaded Data:")
        print(df.head())