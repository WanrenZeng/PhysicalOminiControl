from __future__ import annotations

from collections.abc import Iterable

import torch


class ExponentialMovingAverage:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        decay: float = 0.9999,
        update_after_step: int = 0,
        use_warmup: bool = True,
        inv_gamma: float = 1.0,
        power: float = 0.75,
    ) -> None:
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1).")
        if update_after_step < 0 or inv_gamma <= 0 or power <= 0:
            raise ValueError("EMA warmup settings must be positive.")

        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.use_warmup = bool(use_warmup)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.optimization_step = 0
        self.shadow_params = [parameter.detach().clone() for parameter in parameters]

    def _current_decay(self) -> float:
        if self.optimization_step <= self.update_after_step:
            return 0.0
        if not self.use_warmup:
            return self.decay
        warmup_step = self.optimization_step - self.update_after_step
        warmup_decay = 1 - (1 + warmup_step / self.inv_gamma) ** (-self.power)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError("EMA parameter list changed after initialization.")

        self.optimization_step += 1
        decay = self._current_decay()
        one_minus_decay = 1 - decay
        for shadow_parameter, parameter in zip(self.shadow_params, parameters):
            shadow_parameter.sub_(one_minus_decay * (shadow_parameter - parameter.detach()))

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        for shadow_parameter, parameter in zip(self.shadow_params, parameters):
            parameter.data.copy_(shadow_parameter.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def store(self, parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
        return [parameter.detach().clone() for parameter in parameters]

    @torch.no_grad()
    def restore(self, parameters: Iterable[torch.nn.Parameter], stored_parameters: Iterable[torch.Tensor]) -> None:
        for parameter, stored_parameter in zip(parameters, stored_parameters):
            parameter.data.copy_(stored_parameter.to(device=parameter.device, dtype=parameter.dtype))

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "use_warmup": self.use_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "optimization_step": self.optimization_step,
            "shadow_params": [parameter.detach().cpu() for parameter in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.use_warmup = bool(state_dict["use_warmup"])
        self.inv_gamma = float(state_dict["inv_gamma"])
        self.power = float(state_dict["power"])
        self.optimization_step = int(state_dict["optimization_step"])
        shadow_params = state_dict["shadow_params"]
        if not isinstance(shadow_params, list) or not all(
            isinstance(parameter, torch.Tensor) for parameter in shadow_params
        ):
            raise ValueError("Invalid EMA shadow parameters in checkpoint.")
        self.shadow_params = [parameter.detach().clone() for parameter in shadow_params]


__all__ = ["ExponentialMovingAverage"]
