"""CPU media tensors and execution interruption; model inference lives in extensions."""

import threading

import torch

from comfy.system_memory import virtual_memory_available, virtual_memory_total

torch_version = torch.__version__
total_ram = virtual_memory_total() / (1024 * 1024)


def get_torch_device():
    return torch.device("cpu")


def get_all_torch_devices(exclude_current=False):
    return [] if exclude_current else [get_torch_device()]


def get_torch_device_name(device):
    return "CPU"


def get_total_memory(dev=None, torch_total_too=False):
    total = virtual_memory_total()
    return (total, total) if torch_total_too else total


def get_free_memory(dev=None, torch_free_too=False):
    free = virtual_memory_available()
    return (free, free) if torch_free_too else free


def intermediate_device():
    return get_torch_device()


def intermediate_dtype():
    return torch.float32


def is_oom(error):
    return isinstance(error, (MemoryError, torch.OutOfMemoryError))


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    return weight.to(dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


class InterruptProcessingException(BaseException):
    pass


interrupt_processing_mutex = threading.RLock()
interrupt_processing = False


def interrupt_current_processing(value=True):
    global interrupt_processing
    with interrupt_processing_mutex:
        interrupt_processing = value


def processing_interrupted():
    with interrupt_processing_mutex:
        return interrupt_processing


def throw_exception_if_processing_interrupted():
    global interrupt_processing
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
