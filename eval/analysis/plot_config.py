import os

# Adding latex to the path
# Adding latex to
# os.environ["PATH"] = f"{os.environ['HOME']}/texlive/2024/bin/x86_64-linux:" + os.environ["PATH"]

##############################################
# Plot parameters

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

# mpl.use("pgf") # alternative matplotlib backend using pdflatex

# Set default properties
params = {
    # 'text.usetex': True,  # Use LaTeX to interpret text
    # 'pgf.texsystem': 'pdflatex',  # Enable pgf backend using pdflatex
    # 'pgf.preamble': r'\usepackage{amsmath}',  # Enable LaTeX math for pgf backend
    'text.latex.preamble': r'\usepackage{amsmath}', 
    'font.family': 'serif',  # Specify the LaTeX family font, if desired
    'figure.titlesize': BIGGER_SIZE,   # Title size
    'font.size': MEDIUM_SIZE,         # Default font size
    'lines.markersize': 5,            # Default line marker size
    'axes.titlesize': MEDIUM_SIZE,    # Axes title size
    'axes.labelsize': MEDIUM_SIZE,    # Axes label size
    'legend.fontsize': MEDIUM_SIZE,   # Legend font size 
    'xtick.direction': 'in',          # Ticks inside the panel
    'ytick.direction': 'in',          # Ticks inside the panel
    'xtick.labelsize': SMALL_SIZE,    # x-axis tick label size
    'ytick.labelsize': SMALL_SIZE,    # y-axis tick label size
}

contourLevels = 100  # For contour plots, recommended 100+
colormap = 'bwr'

SAVE_DIR = ''
savefig_format = 'jpg'  # 'pdf', 'jpg', 'png', 'eps' are common options