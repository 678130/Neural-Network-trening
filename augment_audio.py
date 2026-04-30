import argparse
import numpy as np
import librosa
import soundfile as sf
import os

def add_noise(signal, snr_db):
    noise = np.random.randn(len(signal))

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2) + 1e-10

    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(signal_power / (snr_linear * noise_power))

    return signal + noise * scale


def main():
    parser = argparse.ArgumentParser(description="Augment MP3 with noise")
    parser.add_argument("input", help="Input MP3 file")
    parser.add_argument("--snr", type=float, default=None,
                        help="SNR in dB (if not set, random 15–30 dB)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate")

    args = parser.parse_args()

    # Load audio
    print(f"Loading: {args.input}")
    audio, sr = librosa.load(args.input, sr=args.sr)

    # Choose SNR
    if args.snr is None:
        snr_db = np.random.uniform(15, 30)
    else:
        snr_db = args.snr

    print(f"Using SNR: {snr_db:.2f} dB")

    # Apply noise
    augmented = add_noise(audio, snr_db)

    # Normalize to avoid clipping
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-9)

    # Output path
    base = os.path.splitext(args.input)[0]
    output_path = f"{base}_augmented.mp3"

    # Save
    sf.write(output_path, augmented, args.sr)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()