# cython: language_level=3, wraparound=False, boundscheck=False

cimport numpy as np
import numpy as np
import random

cpdef np.ndarray[double] apply_clock_jitters(int jitters_level, np.ndarray[double] trace):
    cdef double average_amplitude
    cdef n = trace.shape[0]
    cdef np.ndarray[double] new_trace = np.empty(n, dtype=trace.dtype)
    i = 0
    put_index = 0

    while i < n and put_index < n:
        new_trace[put_index] = trace[i]
        put_index += 1
        level = random.randint(-jitters_level, jitters_level)
        if level < 0:
            i += -level  # Skip points
        elif level > 0:
            j = 0
            average_amplitude = (trace[i] + trace[i + 1]) / 2 if i + 1 < n else trace[i]
            while j < level and put_index < n:
                new_trace[put_index] = average_amplitude
                put_index += 1
                j += 1
        i += 1

    while put_index < n:
        new_trace[put_index] = 0.0
        put_index += 1

    return new_trace