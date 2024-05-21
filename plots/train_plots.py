import os
import json
import matplotlib.pyplot as plt
from config import *

def read_train_val_results(tests):
    all_results = []

    for t in tests:
        script_directory = os.path.dirname(os.path.realpath(__file__))
        test_file_name = os.path.join(
            script_directory, '..', 'results', t, 'results', 'tr_val_results.json')
        print(test_file_name)
        if os.path.exists(test_file_name):
            with open(test_file_name, 'r') as file:
                try:
                    data = json.load(file)
                    all_results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {test_file_name}: {e}")
        else:
            raise ValueError(f"Test results for {t} don't exist")
    return all_results


def create_line_plots(metrics, data, models_name, configuration, save_plot_prefix="plot"):
    script_directory = os.path.dirname(__file__)
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(10, 7))
        lines = []
        labels = []
        for j, model_data in enumerate(data):
            test_label = f"{models_name[j]} Train {metric[1]}"
            val_label = f"{models_name[j]} Validation {metric[1]}"

            test_values = [epoch[f'training_{metric[0]}']
                           for epoch in model_data]
            val_values = [epoch[f'validation_{metric[0]}']
                          for epoch in model_data]

            line_test, = ax.plot(range(1, len(test_values) + 1),
                                 test_values, marker='o', label=test_label)
            line_val, = ax.plot(range(1, len(val_values) + 1), val_values,
                                marker='o', label=val_label, linestyle='dashed')

            lines.extend([line_test, line_val])
            labels.extend([test_label, val_label])

        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel(metric[1] + " (%)" if metric[0] != "loss" else metric[1], fontsize=14)
        if len(data) == 1:
            ax.set_title(f'{models_name[0]} {metric[1]} Train Results', fontsize=16)
        else:
            ax.set_title(f'{metric[1]} Train Results', fontsize=16)
        ax.legend(lines, labels, loc='best')

        # Add a description under the title
        ax.text(0.5, -0.12, configuration, ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='black')

        # Save the plot to a file
        if not os.path.exists(os.path.join(script_directory, "results")):
            os.makedirs(os.path.join(script_directory, "results"))
        save_path = os.path.join(
            script_directory, "results", f"{save_plot_prefix}_train_{metric[0]}.png")
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")


# ---CONFIGURATIONS---#
test_folders = PATH_MODEL_TO_TEST
# metrics = [('accuracy', 'Accuracy'), ('recall', 'Recall'), ('precision', 'Precision'), ('f1', 'F1'), ('auroc', 'AUROC'), ('loss', 'Cross Entropy Loss')]
metrics = [('accuracy', 'Accuracy'), ('recall', 'Recall'),('f1', 'F1'),('loss', 'Cross Entropy Loss')]
for name in test_folders:
    if name.split("_")[0] == "VideoNet":
        models_name = [name.split("_")[0] + "-" + name.split("_")[1] for name in test_folders]
    else:
        models_name = [name.split("_")[0] for name in test_folders] 

# Read configurations.json file
with open(os.path.join(os.path.dirname(__file__), '..', 'results', test_folders[0], 'configurations.json'), 'r') as file:
    configurations = json.load(file)

batch_size = configurations['batch_size']
configuration = f"batch Size={batch_size}, lr={configurations['learning_rate']}, reg={configurations['reg']}, dropout={configurations['dropout_p']}"


assert len(test_folders) == len(
    models_name), "The number of tests and their name must be of equal length"

data = read_train_val_results(test_folders)
print(data)
create_line_plots(metrics, data, models_name, configuration)
