# Imports
from functools import reduce
import numpy as np
import collections
from typing import List

### ENTER STUDENT CODE BELOW ###
# If required, add further imports here

### ENTER STUDENT CODE ABOVE ###

# Define custom types for tensors and layers
MyTensor = collections.namedtuple("MyTensor", ("idx", "name", "shape", "dtype", "is_const"))
MyLayer = collections.namedtuple("MyLayer", ("idx", "name", "inputs", "outputs"))

"""
Example Usage for FC layer (e.g. y=W'x+b):

>>> input = MyTensor(0, "x", [1, 1024], "int8", False)
>>> weights = MyTensor(1, "W", [10, 1024], "int8", True)
>>> bias = MyTensor(2, "b", [10], "int32", True)
>>> output = MyTensor(3, "y", [1, 10], "int8", False)
>>> fc = MyLayer(0, "FullyConnected", [input.idx, weights.idx, bias.idx], [output.idx])
"""


def estimate_conv2d_macs(in_shape: List[int], kernel_shape: List[int], out_shape: List[int]):
    """Calculate the estimated number of MACS to execute a Conv2D layer.

    Arguments
    ---------
    in_shape : list
        The shape of the NHWC input tensor [batch_size, input_h, input_w, input_c]
    kernel_shape : list
        The shape of the OHWI weight tensor [kernel_oc, kernel_h, kernel_w, kernel_ic]
    out_shape : list
        The shape of the NHWC output tensor [batch_size, output_h, output_w, output_c]


    Returns
    -------
    macs : int
        Estimated number of MAC operations for the given Conv2D layer

    Assumptions
    -----------
    - It is allowed to count #multiplications instead of #macs
    """

    # Hint: not every of the following values is required
    input_n, input_h, input_w, input_c = in_shape
    kernel_oc, kernel_h, kernel_w, kernel_ic = kernel_shape
    output_n, output_h, output_w, output_c = out_shape

    # Assertions
    assert input_n == output_n == 1  # Inference -> batch_size=1
    assert input_c == kernel_ic
    assert output_c == kernel_oc

    macs = 0

    ### ENTER STUDENT CODE BELOW ###
    # Multiplication per output 'pixel'
    mul_per_ouput = kernel_h * kernel_w * kernel_ic
    # Propergate through the ouput 
    macs = mul_per_ouput * (output_n * output_c * output_h * output_w) 
    ### ENTER STUDENT CODE ABOVE ###

    return macs


def estimate_depthwise_conv2d_macs(
    in_shape: List[int], kernel_shape: List[int], out_shape: List[int], channel_mult: int
):
    """Calculate the estimated number of MACS to execute a Depthwise Conv2D layer.

    Arguments
    ---------
    in_shape : list
        The shape of the NHWC input tensor [batch_size, input_h, input_w, input_c]
    kernel_shape : list
        The shape of the weight tensor [1, kernel_h, kernel_w, kernel_oc]
    out_shape : list
        The shape of the NHWC output tensor [batch_size, output_h, output_w, output_c]
    channel_mult : int
        The channel multiplier used to determine the number of output channels.
        See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D


    Returns
    -------
    macs : int
        Estimated number of MAC operations for the given Depthwise Conv2D layer

    Assumptions
    -----------
    - It is allowed to count #multiplications instead of #macs
    """

    # Hint: not every of the following values is required
    input_n, input_h, input_w, input_c = in_shape
    _, kernel_h, kernel_w, kernel_oc = kernel_shape
    output_n, output_h, output_w, output_c = out_shape

    # Assertions
    assert input_n == output_n == 1  # Inference -> batch_size=1
    assert output_c == kernel_oc == input_c * channel_mult

    macs = 0

    ### ENTER STUDENT CODE BELOW ###
    # Muplication per output 'pixel' 
    mul_per_ouput = kernel_h * kernel_w
    # NOTE: it looks different but the channel output here includes the channel mul
    macs = (output_n *  output_c * output_h * output_w) * mul_per_ouput
    ### ENTER STUDENT CODE ABOVE ###

    return macs


def estimate_fully_connected_macs(
    in_shape: List[int], filter_shape: List[int], out_shape: List[int]
):
    """Calculate the estimated number of MACS to execute a Fully Connected layer.

    Arguments
    ---------
    in_shape : list
        The shape of the input tensor [input_h, input_w]
    filter_shape : list
        The shape of the weight tensor [filter_h, filter_w]
    out_shape : list
        The shape of the output tensor [output_h, output_w]


    Returns
    -------
    macs : int
        Estimated number of MAC operations for the given Fully Connected layer

    Assumptions
    -----------
    - It is allowed to count #multiplications instead of #macs
    """

    # Hint: not every of the following values is required
    input_h, input_w = in_shape
    filter_h, filter_w = filter_shape
    output_h, output_w = out_shape

    # Assertions
    assert input_w == filter_h
    assert output_w == filter_w
    assert input_h == output_h

    macs = 0

    ### ENTER STUDENT CODE BELOW ###
    macs = filter_h * filter_w 
    ### ENTER STUDENT CODE ABOVE ###

    return macs


def estimate_rom(tensors: List[MyTensor]):
    """Calculate the estimated number of bytes required to store model weights in ROM.

    Arguments
    ---------
    tensors : list
        The tensors of the processed model (see definition of MyTensor type above)

    Returns
    -------
    rom_bytes : int
        Estimated number of bytes in ROM considering only model weights and biases

    Assumptions
    -----------
    - The considered model will be a fully quantized one in TFLite format
    - TFLite uses different datatypes for the biases.
    - Only constant tensors (weights, biases) contribute to the expected ROM usage (e.g. the model graph, metadata,... can be ignored)
    - A Reshape layer in TFLite has a constant shape tensor, which has to be considered as well.
    """

    rom_bytes = 0

    ### ENTER STUDENT CODE BELOW ###
    # Iterate over each tensor
    for tensor in tensors:
        # Check if the tensor is a constant tensor (this means is stored in ROM)
        if tensor.is_const:
            # Calculate the size of the tensor based on its shape and data type
            tensor_size = np.prod(tensor.shape)  # Total number of elements in the tensor
            if tensor.dtype == 'int8':
                rom_bytes += tensor_size  # Each int8 element is 1 byte
            elif tensor.dtype == 'int32':
                rom_bytes += tensor_size * 4  # Each int32 element is 4 bytes
            elif tensor.dtype == 'float32':
                rom_bytes += tensor_size * 4  # Each float32 element is 4 bytes

    return rom_bytes
    ### ENTER STUDENT CODE ABOVE ###

def estimate_ram(tensors: List[MyTensor], layers: List[MyLayer]):
    """Calculate the estimated number of bytes required to store model tensors in RAM.

    Arguments
    ---------
    tensors : list
        The tensors of the processed model (see definition of MyTensor type above)
    layers : list
        The layers of the processed model (see definition of MyLayer type above)

    Returns
    -------
    ram_bytes : int
        Estimated RAM usage given ideal memory planning (e.g. buffers can be shared to reduce the memory footprint)

    Assumptions
    -----------
    - Only fully-sequential model architectures are considered (e.g. no branches/parallel paths are allowed)
    - Only intermediate tensors (activations) are considered for RAM usage (no temporary workspace buffers are used in the layers)
    - During the operation of a single layer, all of its input and output tensors have to be available
    - In-place operations are not allowed (e.g. the input and output buffer of a layer can not be the same)
    - The input and output tensors of the whole model can also be considered for memory planning

    """

    ram_bytes = 0

    ### ENTER STUDENT CODE BELOW ###    
    # Iterate over each layer to calculate the RAM usage.
    for layer in layers:
        
        input_size = 0
        output_size = 0

        # Find the input tensor by its ID and calculate its size
        for tensor in tensors:
            if tensor.idx == layer.inputs[0] and not tensor.is_const:
                input_size = np.prod(tensor.shape) * np.dtype(tensor.dtype).itemsize
                break 
            
        # Find the output tensor by its ID and calculate its size
        for tensor in tensors:
            if tensor.idx == layer.outputs[0] and not tensor.is_const:
                output_size = np.prod(tensor.shape) * np.dtype(tensor.dtype).itemsize
                break 

        # Calculate the RAM required for the current layer's input and output tensor.
        current_layer_ram = input_size + output_size

        # Update the maximum RAM usage if the current layer requires more than what we've seen before.
        ram_bytes = max(ram_bytes, current_layer_ram)
    ### ENTER STUDENT CODE ABOVE ###
    
    return ram_bytes

def get_dtype_size(dtype: str) -> int:
    """Get the size of the data type in bytes."""
    if dtype == 'int8':
        return 1
    elif dtype == 'int32' or dtype == 'float32':
        return 4
    else:
        raise ValueError(f"Unknown dtype: {dtype}")
