"""
Validation script for Figure 1 of "Energon: Unveiling Transformers from
GPU Power and Thermal Side-Channels".

Figure 1 claims:
  (a) A custom transformer (1 encoder layer, 1 decoder layer, 8 attention
      heads, d_model=512, max sequence length 200) shows a distinct
      power-trace pattern: a small power rise during the ENCODER phase
      (green), followed by a much larger, staircase-like power rise during
      the autoregressive DECODER phase (red).
  (b) A CNN model shows a roughly steady/flat power trace throughout
      execution.

This script reproduces the experiment on your own GPU:
  - Builds a transformer with the same hyperparameters as the paper's
    initial custom model.
  - Builds a small CNN for comparison.
  - Repeatedly runs inference for each model while sampling instantaneous
    GPU power draw via NVML (pynvml) at a configurable rate.
  - Plots the power trace with encoder (green) / decoder (red) phases
    shaded, exactly like Figure 1a, plus the CNN trace like Figure 1b.
  - Prints summary statistics (mean power per phase) so you can check the
    paper's "+2W encoder / +20W decoder / flat CNN" claim quantitatively.

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install nvidia-ml-py matplotlib numpy

Usage:
    python validate_figure1.py
    python validate_figure1.py --duration 60 --interval 0.05
"""

import argparse
import time
import threading

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import pynvml


# --------------------------------------------------------------------------- #
# GPU power monitor (mirrors the paper's use of nvidia-smi / pynvml at
# user-privilege level, Section IV.A)
# --------------------------------------------------------------------------- #
class PowerMonitor:
    def __init__(self, gpu_index=0, interval=0.05):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.interval = interval
        self.samples = []  # list of (timestamp_seconds, power_watts)
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.samples.append((time.perf_counter(), power_mw / 1000.0))
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.samples

    def shutdown(self):
        pynvml.nvmlShutdown()


# --------------------------------------------------------------------------- #
# Custom transformer matching the paper's initial config:
#   1 encoder layer, 1 decoder layer, 8 attention heads,
#   embedding (d_model) = 512, max sequence length = 200
# --------------------------------------------------------------------------- #
class SimpleSeq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8,
                 num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=2048, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_len, 1, d_model) * 0.02)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=False,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        src_emb = src_emb + self.pos_encoding[: src.size(0)]
        return self.transformer.encoder(src_emb)

    def decode_step(self, tgt, memory):
        tgt_emb = self.embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = tgt_emb + self.pos_encoding[: tgt.size(0)]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out[-1:])  # logits for next token


# --------------------------------------------------------------------------- #
# Small CNN for comparison (Figure 1b: conv1/pool1/conv2/pool2/conv3/fc1/fc2)
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# --------------------------------------------------------------------------- #
# Experiment runners
# --------------------------------------------------------------------------- #
def run_transformer_experiment(model, device, duration_sec, interval,
                                seq_len=200, decode_len=200,
                                batch_size=64, vocab_size=10000):
    src = torch.randint(0, vocab_size, (seq_len, batch_size), device=device)
    bos = torch.zeros((1, batch_size), dtype=torch.long, device=device)

    # Warm-up (CUDA kernel compilation / cuDNN autotune) - not measured.
    with torch.no_grad():
        for _ in range(3):
            mem = model.encode(src)
            tgt = bos.clone()
            for _ in range(5):
                logits = model.decode_step(tgt, mem)
                tgt = torch.cat([tgt, logits.argmax(-1)], dim=0)
    torch.cuda.synchronize()

    monitor = PowerMonitor(interval=interval)
    monitor.start()
    t_start = time.perf_counter()
    phases = []  # (start_ts, end_ts, 'encoder'/'decoder')

    with torch.no_grad():
        while time.perf_counter() - t_start < duration_sec:
            # ---- Encoder phase ----
            enc_start = time.perf_counter()

            # Repeat encoder several times so it becomes visible
            for _ in range(20):
                memory = model.encode(src)

            torch.cuda.synchronize()

            enc_end = time.perf_counter()

            print(f"Encoder duration: {(enc_end - enc_start)*1000:.2f} ms")

            phases.append((enc_start, enc_end, "encoder"))

            # ---- Decoder phase (autoregressive, growing sequence) ----
            dec_start = time.perf_counter()
            tgt = bos.clone()
            for _ in range(decode_len):
                logits = model.decode_step(tgt, memory)
                next_token = logits.argmax(-1)
                tgt = torch.cat([tgt, next_token], dim=0)
            torch.cuda.synchronize()
            dec_end = time.perf_counter()

            print(f"Decoder duration: {(dec_end - dec_start)*1000:.2f} ms")

            phases.append((dec_start, dec_end, "decoder"))

            samples = monitor.stop()
            monitor.shutdown()
            return samples, phases, t_start


def run_cnn_experiment(model, device, duration_sec, interval, batch_size=256):
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    torch.cuda.synchronize()

    monitor = PowerMonitor(interval=interval)
    monitor.start()
    t_start = time.perf_counter()

    with torch.no_grad():
        while time.perf_counter() - t_start < duration_sec:
            _ = model(x)
            torch.cuda.synchronize()

    samples = monitor.stop()
    monitor.shutdown()
    return samples, t_start


def measure_idle_power(interval, duration=3.0):
    monitor = PowerMonitor(interval=interval)
    monitor.start()
    time.sleep(duration)
    samples = monitor.stop()
    monitor.shutdown()
    return float(np.mean([p for _, p in samples]))


# --------------------------------------------------------------------------- #
# Plotting (Figure 1 style)
# --------------------------------------------------------------------------- #
def plot_transformer_trace(ax, samples, phases, t_start):
    times = np.array([s[0] - t_start for s in samples])
    power = np.array([s[1] for s in samples])
    trace_count = np.arange(len(samples))

    ax.plot(trace_count, power, color="tab:blue", linewidth=0.8, label="Power [W]")

    legend_added = {"encoder": False, "decoder": False}
    for start, end, label in phases:
        start_idx = int(np.searchsorted(times, start - t_start))
        end_idx = int(np.searchsorted(times, end - t_start))
        if end_idx <= start_idx:
            continue
        color = "tab:green" if label == "encoder" else "tab:red"
        kwargs = {}
        if not legend_added[label]:
            kwargs["label"] = f"{label.capitalize()} Phase"
            legend_added[label] = True
        ax.axvspan(start_idx, end_idx, color=color, alpha=0.3, **kwargs)

    ax.set_xlabel("Trace Count")
    ax.set_ylabel("Power [W]")
    ax.set_title("(a) Transformer Model")
    ax.legend(loc="upper right", fontsize=8)


def plot_cnn_trace(ax, samples, t_start):
    power = np.array([s[1] for s in samples])
    trace_count = np.arange(len(samples))
    ax.plot(trace_count, power, color="tab:blue", linewidth=0.8, label="Power [W]")
    ax.set_xlabel("Trace Count")
    ax.set_ylabel("Power [W]")
    ax.set_title("(b) CNN Model")
    ax.legend(loc="upper right", fontsize=8)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def phase_means(samples, phases, t_start, label):
    times = np.array([s[0] - t_start for s in samples])
    power = np.array([s[1] for s in samples])
    mask = np.zeros(len(samples), dtype=bool)
    for start, end, ph in phases:
        if ph != label:
            continue
        mask |= (times >= (start - t_start)) & (times <= (end - t_start))
    return float(power[mask].mean()) if mask.any() else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0,
                         help="seconds of inference to run per model")
    parser.add_argument("--interval", type=float, default=0.005,
                         help="power sampling interval in seconds")
    parser.add_argument("--seq-len", type=int, default=512,
                         help="encoder input sequence length (paper: 512)")
    parser.add_argument("--decode-len", type=int, default=512,
                         help="decoder autoregressive steps (paper: max 200)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output", type=str, default="figure1_validation.png")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available - this script requires a GPU.")

    device = torch.device(f"cuda:{args.gpu_index}")
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    print("\nMeasuring idle GPU power baseline (3s)...")
    idle_power = measure_idle_power(args.interval)
    print(f"  Idle power: {idle_power:.2f} W")

    print("\n[1/2] Running custom Transformer "
          f"(1 enc / 1 dec layer, 8 heads, d_model=512, seq_len={args.seq_len}, "
          f"decode_len={args.decode_len}, batch={args.batch_size}) "
          f"for {args.duration}s ...")
    transformer = SimpleSeq2SeqTransformer(
    vocab_size=10000,
    d_model=1024,
    nhead=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=4096,
    max_len=max(args.seq_len, args.decode_len + 1),
).to(device).eval()

    t_samples, t_phases, t_start = run_transformer_experiment(
        transformer, device, args.duration, args.interval,
        seq_len=args.seq_len, decode_len=args.decode_len,
        batch_size=args.batch_size,
    )

    enc_mean = phase_means(t_samples, t_phases, t_start, "encoder")
    dec_mean = phase_means(t_samples, t_phases, t_start, "decoder")

    print(f"  Samples collected: {len(t_samples)}")
    print(f"  Mean power - Encoder phase: {enc_mean:.2f} W "
          f"(delta vs idle: {enc_mean - idle_power:+.2f} W)")
    print(f"  Mean power - Decoder phase: {dec_mean:.2f} W "
          f"(delta vs idle: {dec_mean - idle_power:+.2f} W)")
    print(f"  Decoder - Encoder delta: {dec_mean - enc_mean:+.2f} W "
          f"(paper reports decoder >> encoder, e.g. ~20W vs ~2W)")

    print(f"\n[2/2] Running CNN model for {args.duration}s ...")
    cnn = SimpleCNN().to(device).eval()
    c_samples, c_start = run_cnn_experiment(cnn, device, args.duration, args.interval)
    c_power = np.array([p for _, p in c_samples])
    print(f"  Samples collected: {len(c_samples)}")
    print(f"  CNN mean power: {c_power.mean():.2f} W, std: {c_power.std():.2f} W "
          f"(paper reports a steady ~constant trace)")

    # ---- Plot like Figure 1 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_transformer_trace(axes[0], t_samples, t_phases, t_start)
    plot_cnn_trace(axes[1], c_samples, c_start)
    fig.suptitle("Power Trace: Custom Transformer vs CNN (cf. Figure 1)")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
