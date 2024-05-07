# imports
import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)
settings = ['fuse', 'three_channel', 'five_channel', 'original', 'png', 'original_moe', 'png_moe']
setting_labels = ['Earliest fusion', 'Early fusion 3 channels', 'Early fusion 5 channels', 'Late fusion raw', 'Late fusion PNG', 'Latest fusion raw', 'Latest fusion PNG']
classes = ['empty', 'midden', 'mound', 'water']
trials = range(1, 31)
class_eval_metrics = ['precision', 'recall', 'f1']
overall_eval_metrics = ['average_precision', 'average_recall', 'average_f1', 'auc']
overall_eval_metric_labels = ['Precision', 'Recall', 'F1', 'AUC']
class_results = {setting: {eval_metric: {image_class: {} for image_class in classes} for eval_metric in class_eval_metrics} for setting in settings}
overall_results = {setting: {eval_metric: {} for eval_metric in overall_eval_metrics} for setting in settings}
width = 0.1
columnspacing = 1
handletextpad = 0.5
handlelength = 1
fontsize = 10

for setting in settings:
    for trial in trials:
        with open(f'results/{setting}/{setting}_{trial}.json', 'r') as results_json:
            data = json.load(results_json)

            for eval_metric in class_eval_metrics:
                for image_class in classes:
                    class_results[setting][eval_metric][image_class][trial] = data[eval_metric][classes.index(image_class)]

            for eval_metric in overall_eval_metrics:
                overall_results[setting][eval_metric][trial] = data[eval_metric]

class_results_avg = {setting: {eval_metric: {image_class: np.mean([class_results[setting][eval_metric][image_class][trial] for trial in trials]) for image_class in classes} for eval_metric in class_eval_metrics} for setting in settings}
class_results_se = {setting: {eval_metric: {image_class: np.std([class_results[setting][eval_metric][image_class][trial] for trial in trials])/np.sqrt(len(trials)) for image_class in classes} for eval_metric in class_eval_metrics} for setting in settings}

overall_results_avg = {setting: {eval_metric: np.mean([overall_results[setting][eval_metric][trial] for trial in trials]) for eval_metric in overall_eval_metrics} for setting in settings}
overall_results_se = {setting: {eval_metric: np.std([overall_results[setting][eval_metric][trial] for trial in trials])/np.sqrt(len(trials)) for eval_metric in overall_eval_metrics} for setting in settings}

def check_results_complete():
    for setting in settings:
        for trial in trials:
            if not os.path.exists(f'results/{setting}/{setting}_{trial}.json'):
                print(f'{setting.capitalize()} trial {trial} does not exist')

def plot_class_results(image_class):
    class_avg_results_plot = [[class_results_avg[setting][eval_metric][image_class] for eval_metric in class_eval_metrics] for setting in settings]
    class_se_results_plot = [[class_results_se[setting][eval_metric][image_class] for eval_metric in class_eval_metrics] for setting in settings]

    fig, ax = plt.subplots(dpi=300)
    x = np.arange(len(class_eval_metrics))
    multiplier = 0

    for i in range(len(class_avg_results_plot)):
        offset = width * multiplier
        ax.bar(x+offset, class_avg_results_plot[i], width, yerr=2*np.array(class_se_results_plot[i]), label=setting_labels[i])
        print(class_avg_results_plot[i])
        multiplier += 1
    
    ax.set_xticks(x + 3*width)
    ax.set_xticklabels([eval_metric.capitalize() for eval_metric in class_eval_metrics])
    ax.set_title(f'Results for {image_class.capitalize()} Class')
    ax.legend(ncols=1, columnspacing=columnspacing, handletextpad=handletextpad, handlelength=handlelength, fontsize=fontsize)
    plt.savefig(f'figures/{image_class}_results.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f'Plotted {image_class} results')

def plot_overall_results():
    overall_avg_results_plot = [[overall_results_avg[setting][eval_metric] for eval_metric in overall_eval_metrics] for setting in settings]
    overall_se_results_plot = [[overall_results_se[setting][eval_metric] for eval_metric in overall_eval_metrics] for setting in settings]

    fig, ax = plt.subplots(dpi=300)
    x = np.arange(len(overall_eval_metrics))
    multiplier = 0

    for i in range(len(overall_avg_results_plot)):
        offset = width * multiplier
        ax.bar(x+offset, overall_avg_results_plot[i], width, yerr=2*np.array(overall_se_results_plot[i]), label=setting_labels[i])
        print(overall_avg_results_plot[i])
        multiplier += 1
    
    ax.set_xticks(x + 3*width)
    ax.set_xticklabels(overall_eval_metric_labels)
    ax.set_title('Macro-Averaged Results')
    ax.legend(ncols=1, columnspacing=columnspacing, handletextpad=handletextpad, handlelength=handlelength, fontsize=fontsize)
    plt.savefig(f'figures/macro_averaged_results.png', bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f'Plotted macro-averaged results')

if __name__ == '__main__':
    check_results_complete()

    for image_class in classes:
        plot_class_results(image_class)

    plot_overall_results()
