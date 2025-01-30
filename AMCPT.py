# model_modules.py

from typing import Optional, List
import torch
import torch.nn as nn
from torch.autograd import Function
from timm.models.layers import DropPath


class AdaptiveDualFusionBlock(nn.Module):
	"""
	AdaptiveDualFusionBlock Module.
	This block fuses two input sequences (a and b) using bidirectional cross-domain attention and feed-forward layers,
	then outputs a fused representation.

	Args:
		in_dim (int): Input feature dimension.
		num_heads_cross (int): Number of heads in cross-domain attention.
		num_layers_cross (int): Number of cross-domain attention layers.
		transformed_freq_dim (int): Dimension after frequency transformation.
		dropout_rate (float): Dropout rate.
		drop_path_rate (float): Drop path rate.
		seq_len (int): Sequence length.

	Shape:
		- Input a: (batch_size, seq_len, in_dim)
		- Input b: (batch_size, seq_len, in_dim)
		- Output: (batch_size, seq_len, transformed_freq_dim)
	"""
	
	def __init__(
			self,
			in_dim: int,
			num_heads_cross: int,
			num_layers_cross: int,
			transformed_freq_dim: int,
			dropout_rate: float,
			drop_path_rate: float,
			seq_len: int
	):
		super().__init__()
		
		self.in_dim = in_dim
		self.transformed_freq_dim = transformed_freq_dim
		self.num_layers_cross = num_layers_cross
		assert transformed_freq_dim % num_heads_cross == 0, "transformed_freq_dim must be divisible by num_heads_cross"
		activation_fn = nn.GELU()
		
		# Positional embeddings for both modalities
		self.position_embedding_a = nn.Parameter(torch.randn(1, seq_len, in_dim) * 0.02)
		self.position_embedding_b = nn.Parameter(torch.randn(1, seq_len, in_dim) * 0.02)
		
		# Frequency transformations
		self.freq_transform_a = nn.Linear(in_dim, transformed_freq_dim)
		self.freq_transform_b = nn.Linear(in_dim, transformed_freq_dim)
		
		# Cross-domain attention layers: a->b and b->a
		self.cross_attn_layers_a_to_b = nn.ModuleList([
			nn.MultiheadAttention(
				embed_dim=transformed_freq_dim,
				num_heads=num_heads_cross,
				dropout=dropout_rate,
				batch_first=True
			) for _ in range(num_layers_cross)
		])
		self.cross_attn_layers_b_to_a = nn.ModuleList([
			nn.MultiheadAttention(
				embed_dim=transformed_freq_dim,
				num_heads=num_heads_cross,
				dropout=dropout_rate,
				batch_first=True
			) for _ in range(num_layers_cross)
		])
		
		# Feed-forward networks for a and b
		self.ffn_layers_a = nn.ModuleList([
			nn.Sequential(
				nn.Linear(transformed_freq_dim, transformed_freq_dim * 4),
				activation_fn,
				nn.Linear(transformed_freq_dim * 4, transformed_freq_dim)
			) for _ in range(num_layers_cross)
		])
		self.ffn_layers_b = nn.ModuleList([
			nn.Sequential(
				nn.Linear(transformed_freq_dim, transformed_freq_dim * 4),
				activation_fn,
				nn.Linear(transformed_freq_dim * 4, transformed_freq_dim)
			) for _ in range(num_layers_cross)
		])
		
		# Dynamic fusion weight layer
		self.dynamic_weight_mlp = nn.Sequential(
			nn.Linear(transformed_freq_dim * 2, transformed_freq_dim * 8),
			nn.GELU(),
			nn.Linear(transformed_freq_dim * 8, transformed_freq_dim * 2),
			nn.Softmax(dim=-1),
		)
		
		# Normalization layers
		self.cross_attn_norms_a = nn.ModuleList([
			nn.LayerNorm(transformed_freq_dim) for _ in range(num_layers_cross)
		])
		self.cross_attn_norms_b = nn.ModuleList([
			nn.LayerNorm(transformed_freq_dim) for _ in range(num_layers_cross)
		])
		self.ffn_norms_a = nn.ModuleList([
			nn.LayerNorm(transformed_freq_dim) for _ in range(num_layers_cross)
		])
		self.ffn_norms_b = nn.ModuleList([
			nn.LayerNorm(transformed_freq_dim) for _ in range(num_layers_cross)
		])
		
		# DropPath layers
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers_cross)]
		self.drop_paths = nn.ModuleList([
			DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()
			for i in range(num_layers_cross)
		])
		
		self.norm_layer = nn.LayerNorm(transformed_freq_dim)
		self.dropout = nn.Dropout(dropout_rate)
	
	def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
		a += self.position_embedding_a
		b += self.position_embedding_b
		
		a_transformed = self.freq_transform_a(a)
		b_transformed = self.freq_transform_b(b)
		a_original = a_transformed
		b_original = b_transformed
		
		for i in range(self.num_layers_cross):
			# a pathway
			attn_output_a, _ = self.cross_attn_layers_a_to_b[i](query=a_transformed, key=b_original, value=b_original)
			a_transformed = self.cross_attn_norms_a[i](a_transformed + self.drop_paths[i](attn_output_a))
			
			ffn_output_a = self.ffn_layers_a[i](a_transformed)
			a_transformed = self.ffn_norms_a[i](a_transformed + self.drop_paths[i](ffn_output_a))
			
			# b pathway
			attn_output_b, _ = self.cross_attn_layers_b_to_a[i](query=b_transformed, key=a_original, value=a_original)
			b_transformed = self.cross_attn_norms_b[i](b_transformed + self.drop_paths[i](attn_output_b))
			
			ffn_output_b = self.ffn_layers_b[i](b_transformed)
			b_transformed = self.ffn_norms_b[i](b_transformed + self.drop_paths[i](ffn_output_b))
		
		combined = torch.cat([a_transformed, b_transformed], dim=-1)
		weights = self.dynamic_weight_mlp(combined)
		fused = weights[:, :, :self.transformed_freq_dim] * a_transformed + \
		        weights[:, :, self.transformed_freq_dim:] * b_transformed
		
		fused = self.norm_layer(fused)
		fused = self.dropout(fused)
		return fused


def split_feature_patches(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
	"""
	Splits the input tensor into patches along the feature dimension.

	Args:
		tensor (Tensor): Input (batch_size, seq_length, feature_dim).
		patch_size (int): Patch size.

	Returns:
		(batch_size, seq_length, num_patches, patch_size)
	"""
	batch_size, seq_length, feature_dim = tensor.size()
	assert feature_dim % patch_size == 0, f"feature_dim({feature_dim}) must be divisible by patch_size({patch_size})"
	num_patches = feature_dim // patch_size
	patches = tensor.view(batch_size, seq_length, num_patches, patch_size)
	return patches


class CLSEncoder(nn.Module):
	"""
	A single-modal Transformer encoder that uses a CLS token and position embeddings.
	It returns the representation of the CLS token.

	Args:
		embed_dim (int): Embedding dimension.
		num_heads_attn (int): Number of attention heads.
		num_layers_transformer (int): Number of Transformer layers.
		dropout_rate (float): Dropout rate.
		seq_len (int): Sequence length.

	Shape:
		- Input x: (batch_size, seq_length, embed_dim)
		- Output: (batch_size, embed_dim)
	"""
	
	def __init__(
			self,
			embed_dim: int,
			num_heads_attn: int,
			num_layers_transformer: int,
			dropout_rate: float,
			seq_len: int
	):
		super().__init__()

		self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
		self.position_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim) * 0.02)
		
		encoder_layers = []
		for _ in range(num_layers_transformer):
			layer = nn.TransformerEncoderLayer(
				d_model=embed_dim,
				nhead=num_heads_attn,
				dim_feedforward=embed_dim * 4,
				dropout=dropout_rate,
				activation='gelu',
				batch_first=True,
			)
			encoder_layers.append(layer)
		
		self.transformer_encoder = nn.Sequential(*encoder_layers)
		self.layer_norm = nn.LayerNorm(embed_dim)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size = x.size(0)
		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		x += self.position_embedding
		x = self.transformer_encoder(x)
		cls = x[:, 0]
		return cls


class MultiScaleFeatureMerger(nn.Module):
	"""
	Merges a main feature representation with multiple auxiliary features using cross-attention and FFN layers.

	Args:
		feature_dim (int): Feature dimension.
		num_auxiliary (int): Number of auxiliary features.
		num_heads (int): Number of attention heads.
		dropout_rate (float): Dropout rate.

	Shape:
		- main_feature: (batch_size, feature_dim)
		- auxiliary_features: list of (batch_size, feature_dim)
		- Output: (batch_size, feature_dim)
	"""
	
	def __init__(
			self,
			feature_dim: int,
			num_auxiliary: int,
			num_heads: int,
			dropout_rate: float
	):
		super().__init__()
		
		self.fusion_blocks = nn.ModuleList([
			nn.ModuleDict({
				'cross_norm_main': nn.LayerNorm(feature_dim),
				'cross_nor_aux':nn.LayerNorm(feature_dim),
				'cross_attn': nn.MultiheadAttention(
					embed_dim=feature_dim,
					num_heads=num_heads,
					dropout=dropout_rate,
					batch_first=True
				),
				'ffn': nn.Sequential(
					nn.Linear(feature_dim, feature_dim * 4),
					nn.GELU(),
					nn.Dropout(dropout_rate),
					nn.Linear(feature_dim * 4, feature_dim)
				),
				'ffn_norm': nn.LayerNorm(feature_dim)
			}) for _ in range(num_auxiliary)
		])
		self.norm = nn.LayerNorm(feature_dim)
		self.dropout = nn.Dropout(dropout_rate)
	
	def forward(self, main_feature: torch.Tensor, auxiliary_features: List[torch.Tensor]) -> torch.Tensor:
		main_states = self.norm(main_feature.unsqueeze(1))
		
		for aux_feature, fusion_block in zip(auxiliary_features, self.fusion_blocks):
			aux_feature = fusion_block['cross_nor_aux'](aux_feature.unsqueeze(1))
			cross_out, _ = fusion_block['cross_attn'](
				query=main_states,
				key=aux_feature,
				value=aux_feature
			)
			main_states = fusion_block['cross_norm_main'](main_states + self.dropout(cross_out))
			
			ffn_out = fusion_block['ffn'](main_states)
			main_states = fusion_block['ffn_norm'](main_states + self.dropout(ffn_out))
		
		return main_states.squeeze(1)


class PatchAggregator(nn.Module):
	"""
	Aggregates  multiple patch-level features into a single feature via a learnable attention MLP

	Args:
		feature_dim (int): Feature dimension.
		num_patches (int): Number of patches.
	

	Shape:
		- Input: list of patch features [ (batch_size, feature_dim), ... ]
		- Output: (batch_size, feature_dim)
	"""
	
	def __init__(
			self,
			feature_dim: int,
			num_patches: int
	):
		super().__init__()
		
		self.weights = nn.Parameter(torch.ones(num_patches))
		self.norm = nn.LayerNorm(feature_dim)
		self.dynamic_weight_mlp = nn.Sequential(
			nn.Linear(feature_dim, feature_dim * 4),
			nn.GELU(),
			nn.Linear(feature_dim * 4, feature_dim),
			nn.Softmax(dim=1),
		)
	
	def forward(self, patch_features: List[torch.Tensor]) -> torch.Tensor:
		patches_stack = torch.stack(patch_features).permute(1, 0, 2)
		weights = self.dynamic_weight_mlp(patches_stack)
		aggregated = torch.sum(weights * patches_stack, dim=1)
		# return self.norm(aggregated)
		return aggregated


class AMCPT(nn.Module):
	"""
    Adaptive Multi-Domain Cross-Attention Patch Transformer (AMCPT)

    This model fuses EEG features (two sets, typically representing two modalities). It applies a custom cross-modal fusion.
     followed by patch-level transformations and multi-scale feature fusion. Finally, it outputs class logits.

	Shape:
		- EEG input: Tuple of two Tensors [a, b], each (batch_size, seq_len, eeg_in_dim)
		- Output: (batch_size, num_classes)
	"""
	
	def __init__(
			self,
			eeg_in_dim: int = 310,
			num_classes: int = 3,
			num_domain: int = 2,
			dropout_rate: float = 0.1,
			drop_path_rate: float = 0.3,
			domain_generalization: bool = False,
			fusion_num_heads_cross: int = 16,
			fusion_num_layers_cross: int = 4,
			fusion_transformed_freq_dim: int = 256,
			transformer_num_heads_fusion: int = 4,
			transformer_num_layers_fusion: int = 2,
			transformer_num_heads_main: int = 8,
			transformer_num_layers_main: int = 4,
			msff_heads: int = 16,
			seq_len: int = 8
	):
		super().__init__()
		self.eeg_in_dim = eeg_in_dim
		
		# Frequency Linear Transformation and Normalization
		self.patch_sizes = [16, 32, 64]
		
		self.domain_generalization = domain_generalization
		self.domain_classifier = None
		if domain_generalization:
			self.domain_classifier = nn.Sequential(
				nn.Linear(fusion_transformed_freq_dim, fusion_transformed_freq_dim // 2),
				nn.LayerNorm(fusion_transformed_freq_dim // 2),
				nn.GELU(),
				nn.Linear(fusion_transformed_freq_dim // 2, fusion_transformed_freq_dim // 4),
				nn.LayerNorm(fusion_transformed_freq_dim // 4),
				nn.GELU(),
				nn.Linear(fusion_transformed_freq_dim // 4, num_domain),
			)

				
		# Cross Modal Fusion Module (if enabled)
		self.cross_modal_fusion_module = AdaptiveDualFusionBlock(
			in_dim=eeg_in_dim,
			num_heads_cross=fusion_num_heads_cross,
			num_layers_cross=fusion_num_layers_cross,
			transformed_freq_dim=fusion_transformed_freq_dim,
			dropout_rate=dropout_rate,
			drop_path_rate=drop_path_rate,
			seq_len=seq_len
		)
		# Patch input projections
		self.patch_input_projections = nn.ModuleDict({
			str(patch_size): nn.Sequential(
				nn.Linear(patch_size, fusion_transformed_freq_dim // 2),
				nn.GELU(),
				nn.Linear(fusion_transformed_freq_dim // 2, fusion_transformed_freq_dim),
				nn.LayerNorm(fusion_transformed_freq_dim)
			) for patch_size in self.patch_sizes
		})
		
		# Patch-level Transformers
		self.patch_transformers = nn.ModuleDict({
			str(patch_size): CLSEncoder(
				embed_dim=fusion_transformed_freq_dim,
				num_heads_attn=transformer_num_heads_fusion,
				num_layers_transformer=transformer_num_layers_fusion,
				dropout_rate=dropout_rate,
				seq_len=seq_len
			) for patch_size in self.patch_sizes
		})
		
		# Patch feature aggregators
		self.patch_aggregators = nn.ModuleDict({
			str(patch_size): PatchAggregator(
				feature_dim=fusion_transformed_freq_dim,
				num_patches=fusion_transformed_freq_dim // patch_size,
			) for patch_size in self.patch_sizes
		})
		
		# Main Transformer
		self.main_transformer = CLSEncoder(
			embed_dim=fusion_transformed_freq_dim,
			num_heads_attn=transformer_num_heads_main,
			num_layers_transformer=transformer_num_layers_main,
			dropout_rate=dropout_rate,
			seq_len=seq_len
		)
		
		# Multi-Scale Feature Merger
		self.feature_fusion = MultiScaleFeatureMerger(
			feature_dim=fusion_transformed_freq_dim,
			num_auxiliary=len(self.patch_sizes),
			num_heads=msff_heads,
			dropout_rate=dropout_rate
		)
		
		# Classifier
		self.classifier = nn.Sequential(
			nn.Linear(fusion_transformed_freq_dim, fusion_transformed_freq_dim // 2),
			nn.LayerNorm(fusion_transformed_freq_dim // 2),
			nn.GELU(),
			nn.Linear(fusion_transformed_freq_dim // 2, fusion_transformed_freq_dim // 4),
			nn.LayerNorm(fusion_transformed_freq_dim // 4),
			nn.GELU(),
			nn.Linear(fusion_transformed_freq_dim // 4, num_classes)
		)
		
		self.feature_dim = fusion_transformed_freq_dim
		self.num_classes = num_classes
		self.apply(self._init_weights)
	
	def _init_weights(self, m):
		"""
		Initializes the weights of the model.

		Applies specific initialization strategies based on the type of the module:
		- nn.Linear: Xavier uniform initialization for weights and zero for biases.
		- nn.LayerNorm: Ones for weights and zeros for biases.
		- nn.MultiheadAttention: Xavier uniform initialization for Q/K/V and out_proj weights, zero for out_proj biases.

		Args:
			m (nn.Module): The module to initialize.
		"""
		if isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0)
		
		elif isinstance(m, nn.MultiheadAttention):
			if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
				nn.init.xavier_uniform_(m.in_proj_weight)
			if hasattr(m, 'out_proj') and m.out_proj is not None:
				nn.init.xavier_uniform_(m.out_proj.weight)
				if m.out_proj.bias is not None:
					nn.init.constant_(m.out_proj.bias, 0)
	
	def forward_features(self, eeg_inputs: List[torch.Tensor]) -> torch.Tensor:
		"""
		Extract fused features from EEG inputs (and optionally other modalities).

		Args:
			eeg_inputs (List[Tensor]): A list of two EEG feature tensors [a, b], each of shape (batch_size, seq_len, eeg_in_dim).
		Returns:
			Tensor: Fused features of shape (batch_size, fusion_transformed_freq_dim).
		"""
		fused_features = self.cross_modal_fusion_module(eeg_inputs[0], eeg_inputs[1])
		patch_features = []
		for patch_size in self.patch_sizes:
			# Split into patches
			patches = split_feature_patches(fused_features, patch_size)
			_, _, num_patches, _ = patches.size()
			
			patch_cls_tokens = []
			for i in range(num_patches):
				patch = patches[:, :, i, :]  # (batch_size, seq_len, patch_size)
				projected_patch = self.patch_input_projections[str(patch_size)](patch)
				transformed = self.patch_transformers[str(patch_size)](projected_patch)
				patch_cls_tokens.append(transformed)
			
			# Aggregate features from all patches of the current patch size
			aggregated = self.patch_aggregators[str(patch_size)](patch_cls_tokens)
			# aggregated = torch.stack(patch_cls_tokens).mean(dim=0)
			patch_features.append(aggregated)
		
		# Process main feature through the main Transformer
		main_feature = self.main_transformer(fused_features)
		
		# Multi-Scale Feature Merging
		fused_features = self.feature_fusion(main_feature, patch_features)
		
		return fused_features
	
	def forward(self, eeg: Optional[List[torch.Tensor]] = None,  alpha: float = 1.0,
	            reverse: bool = False) -> torch.Tensor:
		"""
		Forward pass.

		Args:
			eeg (List[Tensor], optional): EEG inputs [a, b], each of shape (batch_size, seq_len, eeg_in_dim).
			alpha (float): coefficient of gradient reversal.
            reverse (bool): Whether to enable gradient reversal layer.

		Returns:
			logits (Tensor): Classification logits, shape (batch_size, num_classes).
            domain_pred (Tensor, optional): domain prediction result, shape (batch_size, num_domain), only returned when reverse=True.
		"""
		if eeg is None:
			raise ValueError("EEG input cannot be None.")
		features = self.forward_features(eeg)
		if self.domain_generalization and reverse:
			reverse_features = ReverseLayerF.apply(features, alpha)
			domain_pred = self.domain_classifier(reverse_features)
			logits = self.classifier(features)
			return logits, domain_pred
		else:
			logits = self.classifier(features)
		return logits


class ReverseLayerF(Function):
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)
	
	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha
		return output,None