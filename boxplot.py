import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def draw_plot(data, edge_color, fill_color, ax):
    bp = ax.boxplot(data, patch_artist=True)

    b = True
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       

file_name = 'overlay_results.xlsx'
df = pd.read_excel(file_name)
data = df.values

colors = ['lightblue', 'lightgreen','lightblue', 'lightgreen']
labels = ['Train: Single  - Test: Overlay', 'Train: Single - Test: Single','Train: Single + Overlay - Test: Overlay', 'Train: Single + Overlay - Test: Single']
# ax = df.boxplot(return_type='axes')
fig, ax = plt.subplots()
bp = ax.boxplot(data,
            vert=True,  # vertical box alignment
            patch_artist=True,  # fill with color
            labels=labels)
for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
plt.show()

