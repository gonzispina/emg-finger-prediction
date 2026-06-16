"""
TCN causal compacta para decodificacion EMG -> velocidad / delta de posicion del dedo.

Arquitectura (kernel_size=3, dilations=[1, 3, 9, 27, 81]):

    [B, in_channels, T]
        v
    CausalConvBlock(in_channels -> hidden, dilations=[1, 3])   # residual
        v
    CausalConvBlock(hidden -> hidden, dilations=[9, 27])       # residual
        v
    CausalConv1d(hidden -> hidden, dilation=81) + GELU + Dropout
        v
    Conv1d(hidden -> output_dim, kernel=1)                     # head
        v
    take [:, :, -1]
        v
    [B, output_dim]

Receptive field:
    RF = 1 + (k - 1) * sum(dilations)
       = 1 + 2 * (1 + 3 + 9 + 27 + 81)
       = 243 samples ~ 304 ms a 800 Hz, ~ 380 ms a 640 Hz.

Causalidad: padding = (k - 1) * dilation a ambos lados via Conv1d, luego
Chomp1d para recortar los samples futuros agregados. Resultado: padding
efectivo solo a la izquierda.
"""
from typing import List, Optional, Type

import torch
import torch.nn as nn

# weight_norm migro de torch.nn.utils a torch.nn.utils.parametrizations en
# PyTorch 2.1. Soportamos ambos.
try:
    from torch.nn.utils.parametrizations import weight_norm  # type: ignore
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm  # type: ignore


class Chomp1d(nn.Module):
    """Recorta los ultimos `chomp_size` samples de la dim temporal.

    Usado despues de una Conv1d con padding simetrico para preservar
    causalidad (equivale a padding solo a la izquierda).
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        if chomp_size < 0:
            raise ValueError("chomp_size debe ser >= 0")
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class CausalConv1d(nn.Module):
    """Conv1d causal con WeightNorm opcional.

    padding = (kernel_size - 1) * dilation a ambos lados, luego se chompa
    el mismo numero de samples al final de la dim temporal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.conv = weight_norm(conv) if use_weight_norm else conv
        self.chomp = Chomp1d(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.chomp(self.conv(x))


class CausalConvBlock(nn.Module):
    """Bloque residual con dos CausalConv1d.

    Cada conv tiene su propia dilatacion (lista de 2). Si in_channels difiere
    de out_channels, el residual se proyecta con un Conv1d 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilations: List[int],
        dropout: float,
        activation: Type[nn.Module] = nn.GELU,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if len(dilations) != 2:
            raise ValueError("CausalConvBlock espera exactamente 2 dilataciones")

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilations[0],
                                  use_weight_norm=use_weight_norm)
        self.act1 = activation()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilations[1],
                                  use_weight_norm=use_weight_norm)
        self.act2 = activation()
        self.drop2 = nn.Dropout(dropout)

        self.residual: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.act1(self.conv1(x)))
        out = self.drop2(self.act2(self.conv2(out)))
        return out + self.residual(x)


def compute_receptive_field(kernel_size: int, dilations: List[int]) -> int:
    """RF = 1 + (k - 1) * sum(dilations) para una cadena causal."""
    return 1 + (kernel_size - 1) * sum(dilations)


class EMGTCN(nn.Module):
    """TCN causal para decodificar velocidad/delta del dedo desde EMG.

    Args:
        in_channels: numero de features por timestep (e.g. 2 raw EMG,
            o 4-8 si incluis envelope, derivadas, RMS, etc.).
        hidden_channels: dim del estado interno. Default 32 — chico a
            proposito porque al principio hay poca data.
        output_dim: 1 para velocidad/delta (default), >1 si extendes a
            multi-output (e.g. velocidad + estado de reposo).
        kernel_size: tamaño de kernel temporal. Default 3.
        dropout: dropout despues de cada conv. Default 0.1.
        dilations: 5 enteros. Los primeros 2 forman el bloque residual 1,
            los siguientes 2 el bloque residual 2, el ultimo es la conv
            final antes del head.
        activation: clase de activacion (no instanciada). Default GELU.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        output_dim: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: Optional[List[int]] = None,
        activation: Type[nn.Module] = nn.GELU,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 9, 27, 81]
        if len(dilations) < 2:
            raise ValueError(
                f"Se esperan al menos 2 dilataciones, se recibieron {len(dilations)}"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilations = list(dilations)
        self.receptive_field = compute_receptive_field(kernel_size, dilations)

        # Estructura flexible:
        #   N pares de dilataciones → N bloques residuales.
        #   1 dilatación impar restante (si len es impar) → conv final no-residual.
        n_blocks = len(dilations) // 2
        self.has_final_conv = (len(dilations) % 2 == 1)

        blocks = []
        for i in range(n_blocks):
            in_ch = in_channels if i == 0 else hidden_channels
            blocks.append(CausalConvBlock(
                in_channels=in_ch,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                dilations=dilations[2 * i: 2 * i + 2],
                dropout=dropout,
                activation=activation,
                use_weight_norm=use_weight_norm,
            ))
        self.blocks = nn.ModuleList(blocks)

        if self.has_final_conv:
            self.final_conv = CausalConv1d(
                hidden_channels, hidden_channels, kernel_size, dilations[-1],
                use_weight_norm=use_weight_norm,
            )
            self.final_act = activation()
            self.final_drop = nn.Dropout(dropout)
        else:
            self.final_conv = None
            self.final_act = None
            self.final_drop = None

        self.head = nn.Conv1d(hidden_channels, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """
        x: [batch, in_channels, time]
        horizon: cuantos timesteps finales devolver. 1 = comportamiento clasico.
        return:
            - horizon == 1: [batch, output_dim]              (squeezed sobre T)
            - horizon  > 1: [batch, output_dim, horizon]     (secuencia final)
        """
        if horizon < 1:
            raise ValueError(f"horizon debe ser >= 1, recibido {horizon}")
        h = x
        for block in self.blocks:
            h = block(h)
        if self.has_final_conv:
            h = self.final_drop(self.final_act(self.final_conv(h)))
        h = self.head(h)                                    # [B, output_dim, T]
        if horizon == 1:
            return h[:, :, -1]                              # [B, output_dim]
        return h[:, :, -horizon:]                           # [B, output_dim, horizon]


if __name__ == "__main__":
    in_ch = 2
    dilations = [1, 3, 9, 27, 81]
    rf = compute_receptive_field(kernel_size=3, dilations=dilations)
    print(f"Receptive field: {rf} samples")
    print(f"  ~ {rf / 800 * 1000:5.1f} ms a 800 Hz")
    print(f"  ~ {rf / 640 * 1000:5.1f} ms a 640 Hz (rate observado real)")

    model = EMGTCN(in_channels=in_ch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParametros entrenables: {n_params:,}")

    # Test 1: shape con in_channels=2, T=243
    x = torch.randn(8, in_ch, rf)
    y = model(x)
    print(f"\nTest 1 (raw EMG, 2 canales):")
    print(f"  input  {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (8, 1), f"shape inesperado: {y.shape}"

    # Test 2: in_channels=4 (e.g. raw + envelope)
    model4 = EMGTCN(in_channels=4, output_dim=1)
    y4 = model4(torch.randn(8, 4, rf))
    print(f"\nTest 2 (4 canales, e.g. raw + envelope):")
    print(f"  input  (8, 4, {rf}) -> output {tuple(y4.shape)}")
    assert y4.shape == (8, 1)

    # Test 3: T mas largo que RF
    y_long = model(torch.randn(8, in_ch, rf + 100))
    print(f"\nTest 3 (T={rf + 100} > RF):")
    print(f"  output {tuple(y_long.shape)}")
    assert y_long.shape == (8, 1)

    # Test 4: causalidad. Cambiar inputs FUERA del receptive field del ultimo
    # timestep no debe modificar la salida.
    model.eval()
    with torch.no_grad():
        T = rf + 100
        x_long = torch.randn(1, in_ch, T)
        y_a = model(x_long)
        x_mod = x_long.clone()
        x_mod[:, :, : T - rf] = torch.randn_like(x_mod[:, :, : T - rf])
        y_b = model(x_mod)
        diff = (y_a - y_b).abs().max().item()
    print(f"\nTest 4 (causalidad):")
    print(f"  max |y(x) - y(x_modificado_fuera_RF)| = {diff:.2e}")
    assert diff < 1e-5, "FAIL: el modelo NO es causal"

    # Test 5: cambios DENTRO del RF deben afectar la salida.
    model.eval()
    with torch.no_grad():
        x_in = torch.randn(1, in_ch, rf)
        y_orig = model(x_in)
        x_perturbed = x_in.clone()
        x_perturbed[:, :, -1] += 1.0
        y_perturbed = model(x_perturbed)
        diff_in = (y_orig - y_perturbed).abs().max().item()
    print(f"\nTest 5 (dentro del RF afecta):")
    print(f"  max |y(x) - y(x con ultimo sample perturbado)| = {diff_in:.4f}")
    assert diff_in > 1e-3, "FAIL: el modelo ignora el sample mas reciente"

    print("\nTodos los tests OK")
