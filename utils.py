from tkinter import Tk, filedialog
import os
import numpy as np

def quantize(x, dim_bias):
    bins = list(range(0, 256, 256//dim_bias)) + [256]
    return np.digitize(x, bins, False) - 1

def ask_openfile(filetype=("numpy files","*.npy")):
    root = Tk()
    filepath = filedialog.askopenfilename(title='Select file',
                                            filetypes = [filetype] + [("all files","*.*")])
    filepath = os.path.normpath(filepath)
    root.withdraw()
    return filepath