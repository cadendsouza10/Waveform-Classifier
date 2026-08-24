import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

OUTPUT_DIR = "waveform_dataset"
CLASSES = ["sine", "square", "triangle", "sawtooth"]

IMAGES_PER_CLASS = 500
SAMPLE_POINTS = 1000
IMG_SIZE = (8, 4.8)

# delete old dataset
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)


def generate_waveform(wave_type):
    t = np.linspace(0, 1, SAMPLE_POINTS)

    frequency = random.uniform(1, 10)
    amplitude = random.uniform(0.5, 2.0)
    phase = random.uniform(0, 2 * np.pi)

    if wave_type == "sine":
        y = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    elif wave_type == "square":
        y = amplitude * signal.square(2 * np.pi * frequency * t + phase)

    elif wave_type == "triangle":
        y = amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase,width=0.5)

    elif wave_type == "sawtooth":
        y = amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase)

    if random.random() < 0.40:
        noise_level = random.uniform(0.003, 0.06)
        y = y + np.random.normal(0, noise_level, size = t.shape)

    return t, y, frequency, amplitude


def save_oscilloscope_image(t, y, frequency, amplitude, save_path):
    fig = plt.figure(figsize=IMG_SIZE)

    # full oscilloscope background
    fig.patch.set_facecolor("#b8b08a")

    # plot area leaves room for top and bottom UI
    ax = fig.add_axes([0.04, 0.18, 0.92, 0.68])
    ax.set_facecolor("black")

    waveform_colors = ["#00ff00", "#ffff00", "#00ffff"]
    color = random.choice(waveform_colors)

    ax.plot(t, y, color=color, linewidth=random.uniform(1.7, 3.0))

    ax.grid(True, color="white", linewidth=0.45, alpha=0.25)

    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    margin = random.uniform(0.2, 0.8)
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)
    ax.set_xlim(0, random.uniform(0.75, 1.0))

    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)

    # randomized UI values
    volts_div = random.choice(["100mV/", "200mV/", "500mV/", "1.00V/", "2.00V/"])
    time_div = random.choice(["100µs/", "200µs/", "500µs/", "1.00ms/", "2.00ms/"])
    offset = random.choice(["-50.0mV", "0.00V", "25.0mV", "-120mV", "500mV"])
    sample_rate = random.choice(["1.00GSa/s", "500MSa/s", "250MSa/s"])
    trigger_state = random.choice(["Trig'd", "Ready", "Auto"])

    # top UI bar
    fig.text(0.015, 0.94, "1", color="yellow", fontsize=13, fontweight="bold")
    fig.text(0.09, 0.94, "2", color="lime", fontsize=13, fontweight="bold")
    fig.text(0.17, 0.94, volts_div, color="black", fontsize=12, fontweight="bold")
    fig.text(0.36, 0.94, sample_rate, color="black", fontsize=9)
    fig.text(0.52, 0.94, time_div, color="black", fontsize=12, fontweight="bold")
    fig.text(0.66, 0.94, "0.0s", color="black", fontsize=12)
    fig.text(0.76, 0.94, trigger_state, color="black", fontsize=12, fontweight="bold")
    fig.text(0.88, 0.94, offset, color="black", fontsize=12, fontweight="bold")

    # small channel marker
    fig.text(0.015, 0.50, "2▶", color=color, fontsize=11, fontweight="bold")

    # measurement text
    fig.text(0.42, 0.145, f"Vpp={2 * amplitude:.2f} V", color=color, fontsize=10)

    fig.text(0.58, 0.145, f"Freq={frequency:.2f} kHz", color=color, fontsize=10)

    # bottom UI title
    fig.text(0.015, 0.105, "Trigger Mode and Coupling Menu", color="black", fontsize=12, fontweight="bold")

    # fake buttons
    button_labels = [
        "Mode\nNormal",
        "Coupling\nDC",
        "Noise Rej\n□",
        "HF Reject\n□",
        "Holdoff\n60.000ns"
    ]

    x_positions = [0.03, 0.18, 0.36, 0.54, 0.72]

    for x, label in zip(x_positions, button_labels):
        fig.text(x, 0.035, label, color="black", fontsize=10,
            bbox = dict(
                facecolor="#c9c191",
                edgecolor="white",
                boxstyle="round,pad=0.35"
            )
        )

    # IMPORTANT _NOTE_ TO SELF: do NOT use bbox_inches="tight"
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


for wave_class in CLASSES:
    class_dir = os.path.join(OUTPUT_DIR, wave_class)
    os.makedirs(class_dir, exist_ok=True)

    for i in range(IMAGES_PER_CLASS):
        t, y, frequency, amplitude = generate_waveform(wave_class)

        filename = f"{wave_class}_{i:04d}.png"
        save_path = os.path.join(class_dir, filename)

        save_oscilloscope_image(t, y, frequency, amplitude, save_path)

    print(f"Generated {IMAGES_PER_CLASS} images for {wave_class}")

print("Dataset generation complete.")