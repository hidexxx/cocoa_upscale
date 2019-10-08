import tkinter as tk
from tkinter import ttk


class CocoaInterface(tk.Frame):

    style = ttk.Style()
    style.configure("BW.TLabel", foreground="black", background="white", boarder="black")

    def __init__(self):
        super().__init__()
        self.make_interface()
        self.mainloop()

    def make_interface(self):
        self.aoi_label = ttk.Label(self, text="AOI path", style="BW.TLabel")
        self.aoi_entry = ttk.Entry(self, style="BW.TLabel")
        self.aoi_label.grid(row = 0, column=0)
        self.aoi_entry.grid(row=0, column=1)

        self.start_date_label = ttk.Label(self, text="Start date:", style="BW.TLabel")
        self.start_date_spin = ttk.Entry(self, style="BW.TLabel")
        self.start_date_label.grid(row=1, column=0)
        self.start_date_spin.grid(row=1, column=1)

        self.end_date_label = ttk.Label(self, text="End date:", style="BW.TLabel")
        self.end_date_spin = ttk.Entry(self, style="BW.TLabel")
        self.end_date_label.grid(row=2, column=0)
        self.end_date_spin.grid(row=2, column=1)

        self.epsg_label = ttk.Label(self, text="ESPG number for final projection", style="BW.TLabel")
        self.epsg_entry = ttk.Entry(self, style="BW.TLabel")
        self.epsg_label.grid(row=3, column=0)
        self.epsg_entry.grid(row=3, column=1)

        self.s1_label = ttk.Label(self, text="Path to preprocessed sentinel-1 image", style="BW.TLabel")
        self.s1_entry = ttk.Entry(self, style="BW.TLabel")
        self.s1_label.grid(row=4, column=0)
        self.s1_entry.grid(row=4, column=1)

        self.user_label = ttk.Label(self, text="Scihub username:", style="BW.TLabel")
        self.user_entry = ttk.Entry(self, style="BW.TLabel")
        self.user_label.grid(row=5, column=0)
        self.user_entry.grid(row=5, column=1)

        self.passwd_label = ttk.Label(self, text="Scihub password", style="BW.TLabel")
        self.passwd_entry = ttk.Entry(self, style="BW.TLabel")
        self.passwd_label.grid(row=6, column=0)
        self.passwd_entry.grid(row=6, column=1)

        self.pack()



if __name__=="__main__":
    interface = CocoaInterface()

