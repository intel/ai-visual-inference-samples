# Latency measurement

## Implementation

- `utils/metrics.py` contains logic for measurement of latency of functions.
- It supports a function decorator called `@latency_timer` and calculates the time taken by any function that the decorator is applied to.
- Every function that we would like to calculate the latency for must have the `@latency_timer` decorator.
- The latency is stored in a dictionary (accessed via singleton object) with the function names as keys and a list of runtime durations (each denoting runtime for a frame) as values.
- `calculate_metrics()` function returns the appropriate aggregation of the collected raw runtimes (in milli seconds) per function/frame as described in the following section.

## Invoking latency computation

To aggregate the latency values that were collected, one has to first initialize the `Metrics` singleton like so
```
m = Metrics()
```

Then, to find latency i.e across all frames or per frame or per function one can use the following calls:

| Description | Function call | Returns |
| ----------- | ------------- |------------- |
| Calculates Latency, Per frame latency, fps of function | m.calculate_metrics(\<function name\>)| function_latency, avg_function_latency , function_fps |
| Calculates (Total) Latency of all functions, Per frame latency, fps | m.calculate_metrics()| total_latency, avg_latency, avg_fps |
| Calculates (Total) Latency of all functions, Per frame latency, fps (If number of frames are uniform across all functions) | m.calculate_metrics(\<number of frames\>)| total_latency, avg_latency, avg_fps |

- Latency of all functions is the sum of run times of all functions that have the `@latency_timer` decorator
- Here \<number of frames\> is the number of times the function in question was invoked (assuming it is called once per frame)

## Disabling/Resetting latency timer
To completely clear the dictionary holding all the latency times i.e reset the timer, call the following function:
```
m.reset_timer()
```
This can be useful when skipping warmup frames or calculating average latency across multiple iterations on the same host.

