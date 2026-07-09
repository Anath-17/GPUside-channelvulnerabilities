# GPU Side-Channel Leakage Characterization using CUDA

This project investigates **instruction-dependent** and **data-dependent** power leakage on NVIDIA GPUs by generating CUDA workloads and monitoring GPU power, frequency, temperature, utilization, and memory behavior.

The experiments are inspired by recent GPU side-channel attack research and aim to study how different instruction types and input data influence GPU power consumption.

---

# Hardware

**Laptop**
- ASUS TUF Gaming

**CPU**
- AMD Ryzen 7 7435HS

**GPU**
- NVIDIA GeForce RTX 3050 Laptop GPU

**CUDA Version**
- 12.7

**Driver Version**
- 566.07

---

# Software Requirements

- Windows 11
- CUDA Toolkit
- Visual Studio 2022
- NVIDIA Driver
- Python 3.11+
- Jupyter Notebook

Python Packages

```
pip install pandas matplotlib numpy
```

---

# Project Structure

```
GPU/
│
├── gpu_stress_add.cu          # ADD instruction workload
├── mul_stress.cu              # MUL instruction workload
├── zero_stress.cu             # Data-dependent (all zeros)
├── one_stress.cu              # Data-dependent (all ones)
│
├── gpu_stress.exe
├── mul_stress.exe
├── zero_stress.exe
├── one_stress.exe
│
├── gpu_stress_trace.csv
├── mul_stress_trace.csv
├── zero_stress_trace.csv
├── one_stress_trace.csv
│
└── README.md
```

---

# Experiments

## 1. Instruction-Dependent Leakage

### ADD Workload

Runs billions of floating point addition operations continuously.

Objective

- Stress CUDA ALUs
- Measure GPU power
- Measure frequency
- Measure temperature

Compile

```
nvcc gpu_stress_add.cu -O3 -o gpu_stress.exe
```

Run

```
gpu_stress.exe
```

---

## MUL Workload

Runs floating point multiplication instructions.

Compile

```
nvcc mul_stress.cu -O3 -o mul_stress.exe
```

Run

```
mul_stress.exe
```

---

Compare

- ADD
- MUL

using

- GPU Power
- GPU Frequency
- GPU Temperature

---

# 2. Data-Dependent Leakage

Two workloads are executed.

## Zero Workload

Input array

```
000000000...
```

Compile

```
nvcc zero_stress.cu -O3 -o zero_stress.exe
```

Run

```
zero_stress.exe
```

---

## One Workload

Input array

```
111111111...
```

Compile

```
nvcc one_stress.cu -O3 -o one_stress.exe
```

Run

```
one_stress.exe
```

---

Objective

Compare

- All zeros
- All ones

using GPU sensor measurements.

---

# Logging GPU Sensors

Open PowerShell

Run

```
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used --format=csv -lms 100 > gpu_log.csv
```

Sampling Interval

100 ms

Logged Parameters

- Timestamp
- Temperature
- GPU Utilization
- GPU Power
- Graphics Frequency
- Memory Frequency
- Memory Usage

---

# Jupyter Notebook Analysis

Load CSV

```python
import pandas as pd

df = pd.read_csv(
    "gpu_log.csv",
    encoding="utf-16",
    skipinitialspace=True
)

df.columns = df.columns.str.strip()
```

Clean Data

```python
df["power.draw [W]"] = (
    df["power.draw [W]"]
    .str.extract(r'([\d.]+)')[0]
    .astype(float)
)

df["temperature.gpu"] = (
    df["temperature.gpu"]
    .astype(float)
)

df["utilization.gpu [%]"] = (
    df["utilization.gpu [%]"]
    .str.extract(r'([\d.]+)')[0]
    .astype(float)
)

df["clocks.current.graphics [MHz]"] = (
    df["clocks.current.graphics [MHz]"]
    .str.extract(r'([\d.]+)')[0]
    .astype(float)
)
```

---

# Plot GPU Power

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.plot(df["power.draw [W]"])
plt.title("GPU Power Trace")
plt.xlabel("Samples")
plt.ylabel("Power (W)")
plt.grid(True)
plt.show()
```

---

# Plot GPU Temperature

```python
plt.figure(figsize=(12,4))
plt.plot(df["temperature.gpu"])
plt.title("GPU Temperature Trace")
plt.xlabel("Samples")
plt.ylabel("Temperature (°C)")
plt.grid(True)
plt.show()
```

---

# Plot GPU Frequency

```python
plt.figure(figsize=(12,4))
plt.plot(df["clocks.current.graphics [MHz]"])
plt.title("Graphics Frequency Trace")
plt.xlabel("Samples")
plt.ylabel("Frequency (MHz)")
plt.grid(True)
plt.show()
```

---

# Plot GPU Utilization

```python
plt.figure(figsize=(12,4))
plt.plot(df["utilization.gpu [%]"])
plt.title("GPU Utilization")
plt.xlabel("Samples")
plt.ylabel("Utilization (%)")
plt.grid(True)
plt.show()
```

---

# Expected Results

Instruction-dependent experiment

- ADD workload
- MUL workload

Compare

- Mean Power
- Peak Power
- Temperature
- Frequency

---

Data-dependent experiment

Compare

- Zero workload
- One workload

Observe

- Average Power
- Standard Deviation
- Frequency Stability

---

# Performance Monitoring

Current GPU

- RTX 3050 Laptop GPU

Observed

- GPU Utilization ≈100%
- Graphics Clock ≈1942 MHz
- Temperature ≈65°C
- Power ≈40 W

Power Limit

75 W

No thermal throttling observed during sustained workload.

---

# Future Work

- FP16 workload
- INT32 workload
- Division workload
- Fused Multiply-Add (FMA)
- Tensor Core workloads
- CNN inference power traces
- Transformer encoder/decoder leakage
- Machine learning based instruction classification from power traces

---

# References

1. Instruction-Dependent Leakage on Integrated GPUs

2. Energon: Side-Channel Attacks on GPU-based Deep Learning Models

3. NVIDIA CUDA Programming Guide

4. NVIDIA System Management Interface (nvidia-smi)

---

# Author

**Aniket Nath**

Electronics & Telecommunication Engineering

COEP Technological University

GPU Side-Channel Research Internship
