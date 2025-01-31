from typing import Dict
import numpy as np
from antropy import sample_entropy

from .base import FeatureCalculator


class SampEnCalculator(FeatureCalculator):
	"""
	Sample Entropy calculator.
	"""
	
	def __init__(self):
		pass
	
	def calculate(self, data: dict, order_dict: Dict[str, int] = None) -> np.ndarray:
		"""
		Parameters:
		
		data: EEG data with shape [n_example, n_channel, n_sample].
		order_dict: Embedding dimensions for different frequency bands, default is {"Delta": 2, "Theta": 2, "Alpha": 3, "Beta": 3, "Gamma": 4}.
		Returns:
		
		Computed Sample Entropy data with shape [n_example, total_features].
		"""
		sampen_all = []
		try:
			for band_name, band_data in data.items():
				def compute_sampen(example):
					return np.array([
						sample_entropy(channel_data)
						for channel_data in example
					])
				sampen_per_example = [compute_sampen(example) for example in band_data]
				sampen_all.append(np.stack(sampen_per_example, axis=0))
			
			return np.concatenate(sampen_all, axis=1)
		except Exception as e:
			print(f"Error in SampEnCalculator: {e}")
			raise
