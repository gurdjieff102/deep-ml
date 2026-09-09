import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 1. Add padding
    padded_input = np.pad(
        input_matrix,
        ((padding, padding), (padding, padding)),
        mode='constant'
    )

    # 2. Calculate output size
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    # 3. Slide the kernel
    for i in range(output_height):
        for j in range(output_width):

            # Starting position of the kernel
            row_start = i * stride
            col_start = j * stride

            # Extract the current window
            window = padded_input[
                row_start:row_start + kernel_height,
                col_start:col_start + kernel_width
            ]

            # 4. Element-wise multiply + sum
            output_matrix[i, j] = np.sum(window * kernel)

    return output_matrix