# signal_process/eeg_process/eeg_feature/base.py

from typing import Union
import numpy as np
from abc import ABC, abstractmethod


class PSDResult(dict):
	"""
	PSDResult extends dict to add an attribute indicating the presence of raw data.
	"""
	
	def __init__(self, *args, have_raw=False, **kwargs):
		super().__init__(*args, **kwargs)
		self.have_raw = have_raw
	
	@property
	def raw(self):
		return self.have_raw
	
	def __getattr__(self, item):
		# Allow attribute access to dictionary keys
		try:
			return self[item]
		except KeyError:
			raise AttributeError(f"'PSDResult' object has no attribute '{item}'")


class FeatureCalculator(ABC):
	"""
	Abstract base class for feature calculators.
	"""
	
	@abstractmethod
	def calculate(self, data: Union[np.ndarray, dict]) -> np.ndarray:
		"""
		Abstract method to calculate features.

		Parameters:
		- data: EEG data with shape (num_windows, n_channels, window_samples).
		  If it's a dict, keys correspond to frequency band names, and values to the data.

		Returns:
		- Calculated feature data.
		"""
		pass
