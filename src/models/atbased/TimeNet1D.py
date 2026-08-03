# models/atbased/TimeNet1D.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reversible Instance Norm (opcional).
    Normaliza por instancia y canal a lo largo de T. Si affine=True,
    aplica gamma/beta aprendibles; guarda mean/std para posible denorm.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta",  None)
        # buffers de stats por batch (se rellenan en forward)
        self.register_buffer("_mean", None, persistent=False)
        self.register_buffer("_std",  None, persistent=False)

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        # x: (B, C, T)
        if mode == "norm":
            mean = x.mean(dim=2, keepdim=True)
            std  = x.std(dim=2, keepdim=True).clamp_min(self.eps)
            self._mean, self._std = mean, std
            y = (x - mean) / std
            if self.affine:
                y = y * self.gamma + self.beta
            return y
        elif mode == "denorm":
            if self._mean is None or self._std is None:
                return x
            y = x
            if self.affine:
                y = (y - self.beta) / (self.gamma + self.eps)
            return y * self._std + self._mean
        else:
            raise ValueError("mode debe ser 'norm' o 'denorm'")


class TimeNet1D(nn.Module):
    """
    TimeNet1D (estilo TimeNet original + mejoras prácticas)

    - Encoder: GRU apilado (opción bidireccional) sobre (B, T, C_in)
    - Decoder: GRU sobre consultas temporales aprendibles (no autoregresivo, estable)
      (opcional) Atención Luong dot-product sobre salidas del encoder
    - Cabeza 1x1 → C_out y skip 1x1 opcional (identidad/estabilizador)
    - Normalización de entrada: LayerNorm por paso temporal (default) u opcional RevIN

    I/O:
      in : (B, C_in, T)
      out: (B, C_out, T)
    """
    def __init__(
        self,
        time_steps: int,
        input_channels: int,
        output_channels: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional_encoder: bool = True,
        use_attention: bool = False,
        use_skip: bool = True,
        input_norm: str = "layernorm",   # {"layernorm", "revin", None}
        revin_affine: bool = True,
        **kwargs
    ):
        super().__init__()
        self.T = int(time_steps)
        self.C_in = int(input_channels)
        self.C_out = int(output_channels)
        self.H = int(hidden_size)
        self.L = int(num_layers)
        self.use_attn = bool(use_attention)
        self.use_skip = bool(use_skip)
        self.bidirectional = bool(bidirectional_encoder)

        # -------- Normalización de entrada --------
        self.input_norm_kind = input_norm
        if input_norm == "layernorm":
            # LayerNorm sobre C_in (aplicada a (B,T,C_in))
            self.in_norm = nn.LayerNorm(self.C_in)
        elif input_norm == "revin":
            self.in_norm = RevIN(self.C_in, affine=revin_affine)
        else:
            self.in_norm = None

        # -------- Encoder GRU --------
        enc_hidden = self.H // 2 if self.bidirectional else self.H
        assert enc_hidden > 0, "hidden_size debe ser >= 2 cuando bidirectional=True"

        self.encoder = nn.GRU(
            input_size=self.C_in,
            hidden_size=enc_hidden,
            num_layers=self.L,
            batch_first=True,
            dropout=dropout if self.L > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        enc_out_dim = enc_hidden * (2 if self.bidirectional else 1)

        # Estado inicial del decoder a partir del estado final del encoder
        self.h0_proj = nn.Linear(self.L * enc_out_dim, self.L * self.H)

        # -------- Time queries (no autoregresivo) --------
        self.time_queries = nn.Parameter(torch.empty(self.T, self.H))
        nn.init.normal_(self.time_queries, mean=0.0, std=0.02)

        # -------- Atención (opcional) --------
        if self.use_attn:
            self.ctx_proj = nn.Linear(enc_out_dim, self.H, bias=False)

        # -------- Decoder GRU --------
        self.decoder = nn.GRU(
            input_size=self.H,
            hidden_size=self.H,
            num_layers=self.L,
            batch_first=True,
            dropout=dropout if self.L > 1 else 0.0,
        )

        # -------- Cabeza y skip --------
        self.head = nn.Conv1d(self.H, self.C_out, kernel_size=1)
        self.skip_conv = nn.Conv1d(self.C_in, self.C_out, kernel_size=1) if self.use_skip else None

        # -------- Inicializaciones estables --------
        self._init_weights(enc_out_dim)

    def _init_weights(self, enc_out_dim: int):
        # GRU: weight_ih_* con Xavier, weight_hh_* ortogonal por bloques
        for gru in [self.encoder, self.decoder]:
            for name, p in gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "weight_hh" in name:
                    # Ortogonal por compuerta
                    hidden = p.shape[1]
                    for i in range(0, p.shape[0], hidden):
                        nn.init.orthogonal_(p[i:i+hidden])
                elif "bias" in name:
                    nn.init.zeros_(p)

        nn.init.xavier_uniform_(self.h0_proj.weight)
        nn.init.zeros_(self.h0_proj.bias)
        if self.use_attn:
            nn.init.xavier_uniform_(self.ctx_proj.weight)

        nn.init.kaiming_uniform_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

        if self.skip_conv is not None:
            nn.init.kaiming_uniform_(self.skip_conv.weight, nonlinearity="linear")
            if self.skip_conv.bias is not None:
                nn.init.zeros_(self.skip_conv.bias)

    def _build_decoder_h0(self, h_enc: torch.Tensor) -> torch.Tensor:
        """
        h_enc: (num_layers * num_directions, B, H_enc)
        → (num_layers, B, H)
        """
        LxD, B, Henc = h_enc.shape
        h_flat = h_enc.permute(1, 0, 2).reshape(B, LxD * Henc)   # (B, L*D*Henc)
        h0 = self.h0_proj(h_flat)                                # (B, L*H)
        h0 = h0.view(B, self.L, self.H).permute(1, 0, 2).contiguous()
        return h0

    def _apply_input_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.in_norm is None:
            return x
        if isinstance(self.in_norm, RevIN):
            return self.in_norm(x, mode="norm")
        # LayerNorm espera (B,T,C)
        x_seq = x.permute(0, 2, 1).contiguous()
        x_seq = self.in_norm(x_seq)
        return x_seq.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)  →  y: (B, C_out, T)
        """
        B, C, T = x.shape
        assert T == self.T, f"Se esperaba T={self.T}, llegó T={T}"

        # Normalización de entrada
        x_n = self._apply_input_norm(x)  # (B, C, T) posiblemente normalizado

        # Encoder GRU sobre (B,T,C)
        x_seq = x_n.permute(0, 2, 1).contiguous()        # (B, T, C_in)
        enc_out, h_enc = self.encoder(x_seq)             # enc_out: (B, T, H_enc_out)
        h0_dec = self._build_decoder_h0(h_enc)           # (L, B, H)

        # Decoder con time queries (no autoregresivo)
        q = self.time_queries.unsqueeze(0).expand(B, -1, -1)   # (B, T, H)

        if not self.use_attn:
            dec_out, _ = self.decoder(q, h0_dec)               # (B, T, H)
        else:
            # Paso a paso con atención Luong dot-product
            dec_h = h0_dec
            outs = []
            # enc_out: (B, T, H_enc_out)
            enc_out_T = enc_out.transpose(1, 2)                # (B, H_enc_out, T)
            for t in range(self.T):
                qt = q[:, t:t+1, :]                            # (B, 1, H)
                o_t, dec_h = self.decoder(qt, dec_h)           # (B, 1, H)
                # scores: (B, 1, T)
                scores = torch.bmm(o_t, enc_out.transpose(1, 2))
                alpha = torch.softmax(scores, dim=-1)
                ctx = torch.bmm(alpha, enc_out)                # (B, 1, H_enc_out)
                ctx = self.ctx_proj(ctx)                       # (B, 1, H)
                o_t = o_t + ctx                                # residual simple
                outs.append(o_t)
            dec_out = torch.cat(outs, dim=1)                   # (B, T, H)

        # Proyección a (B, C_out, T)
        dec_out = dec_out.permute(0, 2, 1).contiguous()        # (B, H, T)
        y = self.head(dec_out)                                 # (B, C_out, T)

        # Skip connection (conv 1x1 sobre entrada)
        if self.skip_conv is not None:
            y = y + self.skip_conv(x)

        # Si se usó RevIN, NO hacemos denorm de la salida (objetivo suele estar ya normalizado fuera)
        return y
