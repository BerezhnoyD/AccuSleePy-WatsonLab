import numpy as np
import logging
from UtilityFunctions import utils
from UtilityFunctions.utils import get_spectrograms as get_specs
from UtilityFunctions.utils import compress_spectrograms as compress
from UtilityFunctions.utils import compress_and_replicate_emg as comp_emg


class SpindleDataset(Dataset):
    """
    Custom PyTorch Dataset for EEG/EMG Sleep Spindle Epochs.
    """
    def __init__(self, features_array, labels_array=None):
        """
        Args:
            features_array: Numpy array of shape (num_epochs, 3, 24, 160)
                            Output directly from SpindlePreproc.__call__()
            labels_array: Numpy array of shape (num_epochs,) containing integers 0, 1, 2 
                          (e.g., 0=WAKE, 1=NREM, 2=REM). None if just predicting.
        """
        # Convert numpy arrays to PyTorch tensors
        self.features = torch.tensor(features_array, dtype=torch.float32)
        
        # Ensure the shape is exactly what ExactSPINDLE expects: (Batch, Channels, Freq, Time)
        # If your preprocessing outputs (num_epochs, 3, 160, 24), we transpose it here:
        if self.features.shape[-2:] == (160, 24):
            self.features = self.features.transpose(-2, -1) # Flips to (24, 160)
            
        self.labels = None
        if labels_array is not None:
            self.labels = torch.tensor(labels_array, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        
        return x
		
logger = logging.getLogger('Spindle-AER' + '.' + __name__)


class SpindlePreproc:
    # Initialize the preprocessing class with the parameters required
    def __init__(self, params):
        self.name = params['name']
        self.target_srate = params['target_srate']
        self.stride = params['spectrogram-stride']
        self.time_interval = params['time_interval']
        self.num_neighbors = params['num_neighbors']
        self.eeg_filtering = params['EEG-filtering']
        self.emg_filtering = params['EMG-filtering']

        logger.debug('preprocessing routine created: \n {0}'.format(self))
    # Calling the class will run this body
    def __call__(self, all_signals, signal_header,
                 eeg_idx=np.array([]),
                 eog_idx=np.array([]),
                 emg_idx=np.array([])):

        # Initial setup for the preprocessing and logging
        logger.debug('preprocessing data ...')
        logger.debug(f'eeg indices: {eeg_idx} and emg indices: {emg_idx} '
                     f'vs signal shape: {len(signal_header)}')
        print("Spindle Preproc - ALL SIGNALS SHAPE--RAW",all_signals.shape)
        # Downsample the signals to the target sample rate
        downsampled_signals = []
        print("SIGNAL HEADER ###: ", signal_header)
        for i, sig in enumerate(all_signals):
            srate = signal_header[i]['sample_frequency']
            # if no sample_rate try sample_frequency
            if srate is None:
                srate = signal_header[i]['sample_frequency']
            if srate != self.target_srate:
                downsampled_signals.append(utils.resample(
                    sig[np.newaxis], srate, self.target_srate))
            else:
                if isinstance(sig, list):
                    downsampled_signals.append(sig)
                else:
                    downsampled_signals.append(sig[np.newaxis, ...])
        # Concatenate the downsampled signals
        all_signals = np.concatenate(downsampled_signals)
        print("Spindle Preproc - ALL SIGNALS SHAPE--Re-sampled",all_signals.shape)
        srate = self.target_srate
        stride = self.stride
        win = srate * 2
        print("Sample Rate: ", srate, "Stride-Steps in # Samples: ", stride, "Window Size: ", win, f"Which is sample rate: {srate} * 2")
        # Get the spectrograms for the EEG and EMG signals
        specs = get_specs(all_signals, srate, win, stride, mode='magnitude')
        print("GET-SPEC-Spectrogram shapes: ", specs.shape)

        # comparess the spectrograms by filtering the EEG and EMG signals through the low and high cutoff frequencies
        eeg_spectrograms = compress(specs[eeg_idx, ...], srate, win,
                                    lowcutoff=self.eeg_filtering['lfreq'],
                                    highcutoff=self.eeg_filtering['hfreq'])
        emg_spectrogram = comp_emg(specs[emg_idx, ...], srate, win,
                                   lowcutoff=self.emg_filtering['lfreq'],
                                   highcutoff=self.emg_filtering['hfreq'],
                                   replicate=np.shape(eeg_spectrograms)[1])

        print("COMPRESS-SPEC-Spectrogram EEG Shape: ", eeg_spectrograms.shape)
        print("COMPRESS-SPEC-Spectrogram EMG Shape: ", emg_spectrogram.shape)

        # Concatenate the EEG and EMG spectrograms
        specs = [x for x in [eeg_spectrograms, emg_spectrogram] if x.size > 0]
        specs = np.concatenate(specs, axis=0)
        # Normalize the spectrograms
        specs = utils.normalise_spectrograms(specs)
        print("NORMALIZE-SPEC-Spectrogram shapes: ", specs.shape)

        samples_per_epoch = int(self.time_interval * srate)
        print("Samples per epoch which is time interval * sample rate: ", self.time_interval, "*", srate, "=", samples_per_epoch)
        # This is the results of a spectrogram with this stride
        epoch_size = samples_per_epoch // self.stride
        num_epochs = len(all_signals[0]) // samples_per_epoch
        print("Samples per Epoch is: ", samples_per_epoch)
        print("Epoch Size is calculated by dividing samples per epoch by stride: ", samples_per_epoch, "/", self.stride, "=", epoch_size)
        print("Number of Epochs is calculated by dividing the length of the signal by samples per epoch: ", len(all_signals[0]), "/", samples_per_epoch, "=", num_epochs)
        data = utils.make_epochs(specs, num_epochs, epoch_size)
        data = utils.add_neighbors(data, self.num_neighbors)
        return data
    # Str is used to print the parameters of the class
    def __str__(self):
        return f"""Preprocessing class SpindlePreprocessing replicates the
        exact preprocessing introduced in the paper.
        parameters:
        target_srate: {self.target_srate}
        stride: {self.stride}
        time_interval: {self.time_interval}
        num_neighbors: {self.num_neighbors}
        eeg_filtering: lfreq={self.eeg_filtering['lfreq']},
                       hfreq={self.eeg_filtering['hfreq']}
        emg_filtering: lfreq={self.emg_filtering['lfreq']},
                       hfreq={self.emg_filtering['hfreq']}
        """		
		
		
		
		
# 1. Define your preprocessing parameters (matching your specific study/the paper)
preproc_params = {
    'name': 'SpindleData',
    'target_srate': 100,            # e.g., downsample to 100 Hz
    'spectrogram-stride': 10,       # Step size for the spectrogram
    'time_interval': 4,             # 4-second epochs (as per the paper)
    'num_neighbors': 2,             # 1 target epoch + 2 neighbors = 3 channels
    'EEG-filtering': {'lfreq': 0.5, 'hfreq': 30},
    'EMG-filtering': {'lfreq': 10, 'hfreq': 100}
}

# 2. Instantiate the preprocessor
preprocessor = SpindlePreproc(preproc_params)

# --- ASSUME YOU HAVE LOADED YOUR RAW DATA HERE ---
# all_signals = [raw_eeg_array, raw_emg_array]
# signal_header = [{'sample_frequency': 400}, {'sample_frequency': 400}]
# dummy_labels = np.random.randint(0, 3, size=(num_epochs,)) 

# 3. Run the preprocessing (generates the massive 4D numpy array)
# epoch_data shape will be approx: (num_epochs, 3, 24, 160)
epoch_data = preprocessor(
    all_signals=all_signals, 
    signal_header=signal_header,
    eeg_idx=np.array([0]), # Assuming 1st signal is EEG
    emg_idx=np.array([1])  # Assuming 2nd signal is EMG
)

# 4. Wrap the processed data in our custom PyTorch Dataset
train_dataset = SpindleDataset(features_array=epoch_data, labels_array=dummy_labels)

# 5. Load it into a PyTorch DataLoader
# Batch size of 32 or 64 is typical for this architecture
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=64, 
    shuffle=True,       # Shuffle epochs for training
    drop_last=True      # Drops the last incomplete batch if dataset size isn't divisible by 64
)

# 6. Test the flow into the exact network!
model = ExactSPINDLE(n_classes=3)

# Fetch a single batch from the dataloader
for batch_features, batch_labels in train_loader:
    print(f"Batch Input Shape: {batch_features.shape}") # Expected: [64, 3, 24, 160]
    
    # Pass through the model
    predictions = model(batch_features)
    
    print(f"Batch Output Shape: {predictions.shape}")   # Expected: [64, 3] (Logits for WAKE, NREM, REM)
    break # Just test one batch