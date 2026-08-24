import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "waveform_testingset/square_predict_0001.png"

# OSCILLOSCOPE SETTINGS
VOLTS_PER_DIV = 0.5
TIME_PER_DIV = 500e-6

# standard oscilloscope grid
VERT_DIVS = 8
HORZ_DIVS = 10

def crop_scope_region(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    height, width, _ = img.shape

    top = int(height * 0.08)
    bottom = int(height * 0.92)
    left = int(width * 0.02)
    right = int(width * 0.98)

    cropped = img[top:bottom, left:right]
    return cropped


def extract_waveform(img):

    # conv to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # detect bright pixels
    lower = np.array([15, 80, 120])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    height, width = mask.shape
    x_vals = []
    y_vals = []

    # for each x pos fine the avg y pos of pixels
    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]

        if (len(ys) > 0):
            x_vals.append(x)
            y_vals.append(np.median(ys))

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    if (len(x_vals) < 10):
        raise ValueError("Not enough waveform points detected")
        
    return x_vals, y_vals, mask


def conv_pixels_to_signal(x_pix, y_pix, img_shape):
    height, width = img_shape[:2]

    total_time = HORZ_DIVS * TIME_PER_DIV
    total_voltage = VERT_DIVS * VOLTS_PER_DIV

    sec_per_pix = total_time / width
    volts_per_pix = total_voltage / height

    # conv x pixels to time
    time = x_pix * sec_per_pix

    # conv y pixels to voltage (invert y axis)
    center_y = height / 2
    voltage = (center_y - y_pix) * volts_per_pix

    return time, voltage



def estimate_frequency(time, voltage):

    # remove dc offset
    v = voltage - np.mean(voltage)

    # zero crossings
    zero_crossings = np.where(np.diff(np.sign(v)))[0]

    if (len(zero_crossings) < 3):
        return None, None
    
    crossing_times = time[zero_crossings]

    periods = []

    for i in range(2, len(crossing_times)):
        period = crossing_times[i] - crossing_times[i-2]
        if (period > 0):
            periods.append(period)
        
    if (len(periods) == 0):
        return None, None
        
    avg_period = np.mean(periods)
    frequency = 1 / avg_period

    return frequency, avg_period
    


def estimate_noise(voltage):

    # smooth singal with moving average
    window_size = 15
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(voltage, kernel, mode = 'same')

    noise = voltage - smoothed

    noise_rms = np.sqrt(np.mean(noise**2))

    return noise_rms




def estimate_distortion(time, voltage):

    # simple distortion
    v = voltage - np.mean(voltage)

    dt = np.mean(np.diff(time))

    if dt <= 0:
        return None

    fft_vals = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(len(v), dt)

    magnitudes = np.abs(fft_vals)

    # ignore dc
    magnitudes[0] = 0

    fundamental_index = np.argmax(magnitudes)
    fundamental_mag = magnitudes[fundamental_index]

    if fundamental_mag == 0:
        return None
    
    harmonic_power = 0

    for n in range(2, 6):
        target_freq = n * freqs[fundamental_index]
        harmonic_index = np.argmin(np.abs(freqs - target_freq))

        if harmonic_index < len(magnitudes):
            harmonic_power += magnitudes[harmonic_index] ** 2

    thd = np.sqrt(harmonic_power) / fundamental_mag

    return thd * 100



def analyze_signal():
    img = crop_scope_region(IMAGE_PATH)

    x_pix, y_pix, mask = extract_waveform(img)

    time, voltage = conv_pixels_to_signal(x_pix, y_pix, img.shape)

    frequency, period = estimate_frequency(time, voltage)

    vpp = np.max(voltage) - np.min(voltage)
    amplitude = vpp / 2
    vrms = np.sqrt(np.mean(voltage**2))
    noise_rms = estimate_noise(voltage)
    distortion = estimate_distortion(time, voltage)

    if frequency is not None:
        print(f"Estimated Frequency: {frequency:.2f} Hz")
        print(f"Estimated Period: {period:.6f} s")
    else:
        print("Estimated Frequency: Could not calculate")

    print(f"Estimated Amplitude: {amplitude:.2f} V")
    print(f"Estimated Vrms: {vrms:.2f} V")
    print(f"Estimated Noise RMS: {noise_rms:.3f} V")

    if distortion is not None:
        print(f"Estimated Distortion (THD%): {distortion:.2f} %")
    else:
        print("Estimated Distortion: Could not calculate")

    # show waveform mask
    cv2.imshow("Waveform Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plot extracted signal
    plt.plot(time, voltage)
    plt.title("Extracted Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()



analyze_signal()