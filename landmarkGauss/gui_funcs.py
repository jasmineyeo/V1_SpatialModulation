# -*- coding: utf-8 -*-
"""
fm2p/utils/gui_funcs.py

DMM, March 2025
"""


import tkinter as tk
from tkinter import filedialog


def select_file(title, filetypes):

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )

    return file_path


def select_directory(title):

    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(
        title=title,
    )

    return directory_path


def get_string_input(title):

    root = tk.Tk()
    label = tk.Label(root, text=title)
    root.minsize(width=300, height=20)
    root.title(title)
    label.pack()
    entry = tk.Entry(root)
    entry.pack()
    user_input = None

    def retrieve_input():
        nonlocal user_input
        user_input = entry.get()
        root.destroy()
        
    button = tk.Button(root, text='Enter', command=retrieve_input)
    button.pack()

    root.bind("<Return>", lambda event: retrieve_input())

    entry.focus_set()
    root.lift()
    root.attributes('-topmost', True)
    root.mainloop()


    return user_input

