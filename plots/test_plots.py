import os
import json
import matplotlib.pyplot as plt
from config import *


def read_test_results(tests):
    all_results = []

    for t in tests:
        script_directory = os.path.dirname(os.path.realpath(__file__))
        test_file_name = os.path.join(
            script_directory, '..', 'results', t, 'results', 'test_results.json')
        print(test_file_name)
        if os.path.exists(test_file_name):
            with open(test_file_name, 'r') as file:
                try:
                    data = json.load(file)[0]
                    all_results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {test_file_name}: {e}")
        else:
            raise ValueError(f"Test results for {t} don't exist")
    return all_results


def create_plots(metrics, data, models_name, configuration, save_plot_prefix="plot"):
    script_directory = os.path.dirname(__file__)
    values = [[d["test_"+metric[0]] for d in data] for metric in metrics]
    colors = ["#7b00c2", "#189e00"]
    legend = ["Without Overlap", "With Overlap"]

    for i, metric in enumerate(metrics):
        _, ax = plt.subplots(figsize=(12, 10))
        for j in range(len(models_name)):
            if j < 2:
                ax.bar(j, values[i][j], color=colors[0], width=0.9)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels([f"{models_name[j]}" for j in range(len(data))], fontsize=12)
                ax.set_ylabel(f"{metric[1]} (%)" if metric[0] != "test_loss" else metric[1], fontsize=14)
                ax.set_title(f'{metric[1]} Test Results', fontsize=16)
            else:
                ax.bar(j, values[i][j], color=colors[1], width=0.9)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels([f"{models_name[j]}" for j in range(len(data))], fontsize=12)
                ax.set_ylabel(f"{metric[1]} (%)" if metric[0] != "test_loss" else metric[1], fontsize=14)
                ax.set_title(f'{metric[1]} Test Results', fontsize=16)


        # Add a legend based on the colors
        # Without overlap: #7b00c2, with overlap: #189e00
        ax.legend(legend, loc='best')          

        # Add a description under the title
        ax.text(0.5, -0.1, configuration, ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='black')

        # Save the plot to a file
        if not os.path.exists(os.path.join(script_directory, "results")):
            os.makedirs(os.path.join(script_directory, "results"))
        save_path = os.path.join(
            script_directory, "results", f"video_{save_plot_prefix}_test_{metric[0]}.png")
        plt.savefig(save_path)
        print(f"Plot saved as {save_path}")

    # for i, metric in enumerate(metrics):
    #     _, ax = plt.subplots(figsize=(6, 6))
    #     # Generate random colors for each bar
    #     ax.bar(range(len(data)), values[i], color=colors, width=0.9)
    #     ax.set_xticks(range(len(data)))
    #     ax.set_xticklabels([f"{models_name[j]}" for j in range(len(data))], fontsize=14)
    #     ax.set_ylabel(f"{metric[1]} (%)" if metric[0] != "test_loss" else metric[1], fontsize=14)
    #     ax.set_title(f'{metric[1]} Test Results', fontsize=16)

    #     # Add a description under the title
    #     ax.text(0.5, -0.1, configuration, ha='center', va='center',
    #             transform=ax.transAxes, fontsize=11, color='black')

    #     # Save the plot to a file
    #     if not os.path.exists(os.path.join(script_directory, "results")):
    #         os.makedirs(os.path.join(script_directory, "results"))
    #     save_path = os.path.join(
    #         script_directory, "results", f"{save_plot_prefix}_test_{metric[0]}.png")
    #     plt.savefig(save_path)
    #     print(f"Plot saved as {save_path}")


# ---CONFIGURATIONS---#
test_folders = PATH_MODEL_TO_TEST
# metrics = [('accuracy', 'Accuracy'), ('recall', 'Recall'), ('precision', 'Precision'), ('f1', 'F1'), ('auroc', 'AUROC'), ('loss', 'Cross Entropy Loss')]
metrics = [('accuracy', 'Accuracy'), ('recall', 'Recall'),('loss', 'Cross Entropy Loss')]
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
    models_name), "The number of tests and their name must be of equal lenght"

data = read_test_results(test_folders)
create_plots(metrics, data, models_name, configuration)
