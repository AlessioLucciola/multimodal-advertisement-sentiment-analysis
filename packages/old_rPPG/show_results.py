import numpy as np

pulse_path = 'pulse.npy'
hrs_path = 'hrs.npy'
fft_spec_path = 'fft_spec.npy'

def main():
    pulse = np.load(pulse_path)
    hrs = np.load(hrs_path)
    fft_spec = np.load(fft_spec_path)
    
    print(f'Pulse: {pulse}')
    print(f'Heart Rate: {hrs}')
    print(f'FFT Spectrum: {fft_spec}')
    
    print('Done')


if __name__ == '__main__':
    main()
