import matplotlib as mpl
import numpy as np

# Centralized plotting defaults for the project
# Adjust figure size, DPI and font settings here
mpl.rcParams.update({
    'figure.figsize': (4,3),
    'figure.dpi': 150,
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'svg.fonttype': 'none',
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Project color palettes (slow / fast muscle families)
palette_cont_slow = tuple(np.flip((
    '#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15'
)))
palette_cont_fast = tuple(np.flip((
    '#eff3ff', '#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c'
)))

# Shared line styles (used across scripts)
ls_styles = ('-', '--', ':', '-.', (0, (3, 1, 1, 1)))
