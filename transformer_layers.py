import torch.nn as nn
import torch
import numpy as np


def create_look_ahead_mask(size):
    """Creates a lookahead mask for autoregressive masking."""
    mask = np.triu(np.ones((size, size), np.float32), 1)
    return torch.Tensor(mask)


class StochasticDepth(nn.Module):
    """Creates a stochastic depth layer."""

    def __init__(self, stochastic_depth_drop_rate):
        """Initializes a stochastic depth layer.

        Args:
          stochastic_depth_drop_rate: A `float` of drop rate.
          name: Name of the layer.

        Returns:
          A output `tf.Tensor` of which should have the same shape as input.
        """
        super().__init__()
        self._drop_rate = stochastic_depth_drop_rate

    def forward(self, inputs):
        if not self.training or self._drop_rate == 0.:
            return inputs
        keep_prob = 1.0 - self._drop_rate
        batch_size = inputs.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand_like(
            [batch_size] + [1] * (inputs.shape.rank - 1), dtype=inputs.dtype)
        binary_tensor = torch.floor(random_tensor)
        output = torch.div(inputs, keep_prob) * binary_tensor
        return output


class MLP(nn.Module):
    """MLP head for transformer."""

    def __init__(self, n_channel,expansion_rate, act, dropout_rate):
        super().__init__()
        self._expansion_rate = expansion_rate
        self._act = act
        self._dropout_rate = dropout_rate
        self._fc1 = nn.Linear(
            n_channel,
            self._expansion_rate * n_channel)
        self.act1 = self._act()
        self._fc2 = nn.Linear(
            self._expansion_rate * n_channel,
            n_channel)
        self.act2 = self._act()
        self._drop = nn.Dropout(self._dropout_rate)

    def forward(self, features):
        """Forward pass."""
        features = self.act1(self._fc1(features))
        features = self._drop(features)
        features = self.act2(self._fc2(features))
        features = self._drop(features)
        return features


class TransformerBlock(nn.Module):
    """Transformer block that is similar to the Swin encoder block.

    However, an important difference is that we _do not_ shift the windows
    for the second Attention layer. Instead, we _feed the encoder outputs_
    as Keys and Values. This allows for autoregressive applications.

    If `style == "encoder"`, no autoregression is happening.

    Also, this class operates on windowed tensor, see `call` docstring.
    """

    def __init__(
        self,
        *,
        d_model,
        seq_len,
        num_head = 4,
        mlp_expansion = 4,
        mlp_act = nn.GELU,
        drop_out_rate = 0.1,
        drop_path_rate = 0.1,
        style = "decoder",
    ):
        super().__init__()
        self._style = style
        if style == "decoder":
            self.look_ahead_mask = create_look_ahead_mask(seq_len)
        elif style == "encoder":
            self.look_ahead_mask = None
        else:
            raise ValueError(f"Invalid style: {style}")

        # self._norm1a = nn.LayerNorm(
        #     axis=-1, epsilon=1e-5, name="mhsa_normalization1")
        self._norm1a = nn.LayerNorm(d_model)
        # self._norm1b = tf.keras.layers.LayerNormalization(
        #     axis=-1, epsilon=1e-5, name="ffn_normalization1")
        self._norm1b = nn.LayerNorm(d_model,eps=1e-5)

        # self._norm2a = tf.keras.layers.LayerNormalization(
        #     axis=-1, epsilon=1e-5, name="mhsa_normalization2")
        self._norm2a = nn.LayerNorm(d_model, eps=1e-5)
        # self._norm2b = tf.keras.layers.LayerNormalization(
        #     axis=-1, epsilon=1e-5, name="ffn_normalization2")
        self._norm2b = nn.LayerNorm(d_model, eps=1e-5)
        self._attn1 = nn.MultiheadAttention(
            d_model,
            num_head,
            dropout=drop_out_rate
        )

        self._attn2 = nn.MultiheadAttention(
            d_model,
            num_head,
            dropout=drop_out_rate
        )

        self._mlp1 = MLP(
            d_model,
            expansion_rate=mlp_expansion,
            act=mlp_act,
            dropout_rate=drop_out_rate)
        self._mlp2 = MLP(
            d_model,
            expansion_rate=mlp_expansion,
            act=mlp_act,
            dropout_rate=drop_out_rate)

        # No weights, so we share for both blocks.
        self._drop_path = StochasticDepth(drop_path_rate)

    def forward(self, features, enc_output):
        if enc_output is None:
            if self._style == "decoder":
                raise ValueError("Need `enc_output` when running decoder.")
        else:
            assert enc_output.shape[0] == features.shape[0] and enc_output.shape[2] == features.shape[2]

        # First Block ---
        shortcut = features
        features = self._norm1a(features)
        # Masked self-attention.
        features = features.permute(1, 0, 2)  # NLD -> LND
        features, _ = self._attn1(
            value=features,
            key=features,
            query=features,
            attn_mask=self.look_ahead_mask)
        features = features.permute(1, 0, 2)  # LND -> NLD

        assert features.shape == shortcut.shape
        features = shortcut + self._drop_path(features)

        features = features + self._drop_path(
            self._mlp1(self._norm1b(features)))

        # Second Block ---
        shortcut = features
        features = self._norm2a(features)
        # Unmasked "lookup" into enc_output, no need for mask.

        features = features.permute(1, 0, 2)  # NLD -> LND
        if enc_output is not None:
            enc_output = enc_output.permute(1, 0, 2)  # NLD -> LND
        features, _ = self._attn2(  # pytype: disable=wrong-arg-types  # dynamic-method-lookup
            value=enc_output if enc_output is not None else features,
            key=enc_output if enc_output is not None else features,
            query=features,
            attn_mask=None)
        features = features.permute(1, 0, 2)  # LND -> NLD

        features = shortcut + self._drop_path(features)
        output = features + self._drop_path(
            self._mlp2(self._norm2b(features)))

        return output


class Transformer(nn.Module):
    """A stack of transformer blocks, useable for encoding or decoding."""

    def __init__(
        self,
        is_decoder,
        num_layers = 4,
        d_model = 192,
        seq_len = 16,
        num_head = 4,
        mlp_expansion = 4,
        drop_out = 0.1
    ):
        super().__init__()
        self.is_decoder = is_decoder

        # Use a plain list here since we have to pass the enc_output to each.
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
              TransformerBlock(
                  d_model=d_model,
                  seq_len=seq_len,
                  num_head=num_head,
                  mlp_expansion=mlp_expansion,
                  drop_out_rate=drop_out,
                  drop_path_rate=drop_out,
                  style="decoder" if is_decoder else "encoder",
              ))

    def forward(
      self, latent, enc_output
    ):
        """Forward pass.

        For decoder, this predicts distribution of `latent` given `enc_output`.

        We assume that `latent` has already been embedded in a d_model-dimensional
        space.

        Args:
          latent: (B', seq_len, C) latent.
          enc_output: (B', seq_len_enc, C) result of concatenated encode output.
          training: Whether we are training.

        Returns:
          Decoder output of shape (B', seq_len, C).
        """
        assert len(latent.shape) == 3, latent.shape
        if enc_output is not None:
            assert latent.shape[-1] == enc_output.shape[-1], (latent.shape,
                                                            enc_output.shape)
        for layer in self.layers:
            latent = layer(features=latent, enc_output=enc_output)
        return latent

