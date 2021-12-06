# cython: language_level=3, wraparound=False, boundscheck=False

cimport numpy as np
import numpy as np
import random

cpdef np.ndarray[double] apply_random_delay_interrupts(int a, int b, double rdi_probability,
                                                       double rdi_amplitude, np.ndarray[double] trace):
    cdef n = trace.shape[0]
    cdef np.ndarray[double] new_trace = np.empty(n, dtype=trace.dtype)
    put_index = 0
    i = 0

    while i < n and put_index < n:
        new_trace[put_index] = trace[i]
        put_index += 1
        if random.random() <= rdi_probability:
            m = random.randint(0, a - b)
            rdi_num = random.randint(m, m + b)
            j = 0
            while j < rdi_num and put_index + 2 < n:
                new_trace[put_index] = trace[i]
                new_trace[put_index + 1] = trace[i] + rdi_amplitude
                new_trace[put_index + 2] = trace[i + 1]
                put_index += 3
                j += 1
        i += 1

    while put_index < n:
        new_trace[put_index] = 0.0
        put_index += 1

    return new_trace
