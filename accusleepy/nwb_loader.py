import os
import sys

import h5py
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class ChannelSelectorDialog(QDialog):
    """Choose a time slice and EEG/EMG channels from an NWB file."""

    def __init__(self, nwb_path, total_duration, fs, n_channels, channel_names=None):
        super().__init__()
        self.setWindowTitle("Select Data to Load")
        self.resize(500, 600)

        self.result_data = None
        self.nwb_path = nwb_path
        self.fs = fs
        self.total_duration = total_duration

        layout = QVBoxLayout()
        self.setLayout(layout)

        info_group = QGroupBox("File Info")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f"File: {os.path.basename(nwb_path)}"))
        info_layout.addWidget(QLabel(f"Duration: {total_duration:.2f} sec"))
        info_layout.addWidget(QLabel(f"Rate: {fs} Hz"))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

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

        lists_layout = QHBoxLayout()

        eeg_group = QGroupBox("Select ONE EEG Channel")
        eeg_layout = QVBoxLayout()
        self.list_eeg = QListWidget()
        self.list_eeg.setSelectionMode(QAbstractItemView.SingleSelection)
        eeg_layout.addWidget(self.list_eeg)
        eeg_group.setLayout(eeg_layout)
        lists_layout.addWidget(eeg_group)

        emg_group = QGroupBox("Select ONE EMG Channel")
        emg_layout = QVBoxLayout()
        self.list_emg = QListWidget()
        self.list_emg.setSelectionMode(QAbstractItemView.SingleSelection)
        emg_layout.addWidget(self.list_emg)
        emg_group.setLayout(emg_layout)
        lists_layout.addWidget(emg_group)

        layout.addLayout(lists_layout)

        for i in range(n_channels):
            name = channel_names[i] if channel_names and i < len(channel_names) else f"Channel {i}"

            item_eeg = QListWidgetItem(name)
            item_eeg.setData(Qt.ItemDataRole.UserRole, i)
            self.list_eeg.addItem(item_eeg)

            item_emg = QListWidgetItem(name)
            item_emg.setData(Qt.ItemDataRole.UserRole, i)
            self.list_emg.addItem(item_emg)

        if n_channels > 0:
            self.list_eeg.item(0).setSelected(True)
            self.list_emg.item(1 if n_channels > 1 else 0).setSelected(True)

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
            start = float(self.txt_start.text())
            dur = float(self.txt_duration.text())

            if start < 0 or start >= self.total_duration:
                raise ValueError("Invalid Start Time")
            if dur <= 0:
                raise ValueError("Duration must be positive")
            if start + dur > self.total_duration:
                QMessageBox.warning(self, "Warning", "Truncating duration to file end.")
                dur = self.total_duration - start

            eeg_items = self.list_eeg.selectedItems()
            emg_items = self.list_emg.selectedItems()

            if not eeg_items:
                raise ValueError("Please select one EEG channel.")
            if not emg_items:
                raise ValueError("Please select one EMG channel.")

            eeg_idx = eeg_items[0].data(Qt.ItemDataRole.UserRole)
            eeg_name = eeg_items[0].text()
            emg_idx = emg_items[0].data(Qt.ItemDataRole.UserRole)
            emg_name = emg_items[0].text()

            indices = [eeg_idx, emg_idx]
            names = ["eeg", "emg"]
            data_matrix = self._read_nwb_chunk(indices, start, dur)
            time_vector = np.linspace(start, start + dur, data_matrix.shape[0], endpoint=False)
            df = pd.DataFrame(data_matrix, columns=names)
            df.insert(0, "Time", time_vector)
            self.result_data = df
            self.accept()
        except Exception as exc:  # pragma: no cover - GUI error path
            QMessageBox.critical(self, "Error", str(exc))

    def _read_nwb_chunk(self, indices, start, duration):
        with h5py.File(self.nwb_path, "r") as f:
            dset = None
            if "acquisition" in f:
                acq = f["acquisition"]
                if "ElectricalSeries" in acq:
                    dset = acq["ElectricalSeries"]["data"]
                elif len(acq.keys()) > 0:
                    dset = acq[list(acq.keys())[0]]["data"]

            if dset is None:
                raise KeyError("Data not found.")

            start_samp = int(start * self.fs)
            end_samp = start_samp + int(duration * self.fs)
            sorted_idx = sorted({int(i) for i in indices})
            data_chunk = dset[start_samp:end_samp, sorted_idx]
            val_to_col = {val: i for i, val in enumerate(sorted_idx)}

            final_data = np.zeros((data_chunk.shape[0], len(indices)))
            for i, target_idx in enumerate(indices):
                col_in_chunk = val_to_col[int(target_idx)]
                final_data[:, i] = data_chunk[:, col_in_chunk]

            return final_data


def load_nwb(filename: str) -> pd.DataFrame | None:
    """Load an NWB file and allow the user to pick EEG and EMG channels."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None

    try:
        with h5py.File(filename, "r") as f:
            dset = None
            series = None
            if "acquisition" in f:
                acq = f["acquisition"]
                if "ElectricalSeries" in acq:
                    series = acq["ElectricalSeries"]
                elif len(acq.keys()) > 0:
                    series = acq[list(acq.keys())[0]]

            if series is None:
                raise ValueError("Invalid NWB.")

            dset = series["data"]
            shape = dset.shape

            fs = 250.0
            if "starting_time" in series:
                fs = series["starting_time"].attrs.get("rate", fs)
            elif "rate" in series:
                fs = series["rate"][()]

            duration = shape[0] / fs
            n_channels = shape[1]
            names = [f"Ch {i}" for i in range(n_channels)]
    except Exception as exc:  # pragma: no cover - NWB compatibility path
        print(f"Failed to read metadata: {exc}")
        return None

    dialog = ChannelSelectorDialog(filename, duration, fs, n_channels, names)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.result_data
    return None


__all__ = ["ChannelSelectorDialog", "load_nwb"]

