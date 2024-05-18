import matplotlib.pyplot as plt
import numpy as np

def plot_signal(data: np.ndarray, title: str) -> None:
    x_time = np.linspace(0, data.shape[0], num=data.shape[0])
    plt.figure(figsize=(10,5))
    plt.plot(x_time, data, color='b', linewidth=1)
    plt.title(title)
    plt.legend(['Predictions', ])
    plt.xlabel('Time (s)');
    plt.savefig(title)
