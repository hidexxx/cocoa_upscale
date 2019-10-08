from tkinter import ttk
import configparser
from build_cocoa_map import build_cocoa_map

class CocoaInterface(ttk.Frame):

    style = ttk.Style()
    style.configure("BW.TLabel", foreground="black", background="white", boarder="black")
    style.configure("BW.TEntry", foreground="black", padding=6)
    style.configure("BW.Frame", foreground="black", background="white")
    style.configure("BW.TButton", foreground="black", background="white")

    def __init__(self):
        super().__init__()
        self.make_interface()
        self.mainloop()

    def make_interface(self):
        """Creates and lays out the elements of the interface."""

        self.aoi_label = ttk.Label(self, text="AOI path", style="BW.TLabel")
        self.aoi_entry = ttk.Entry(self, style="BW.TEntry")
        self.aoi_label.grid(row = 0, column=0, sticky="W")
        self.aoi_entry.grid(row=0, column=1)

        self.start_date_label = ttk.Label(self, text="Start date:", style="BW.TLabel")
        self.start_date_spin = ttk.Entry(self, style="BW.TEntry")
        self.start_date_label.grid(row=1, column=0, sticky="W")
        self.start_date_spin.grid(row=1, column=1)

        self.end_date_label = ttk.Label(self, text="End date:", style="BW.TLabel")
        self.end_date_spin = ttk.Entry(self, style="BW.TEntry")
        self.end_date_label.grid(row=2, column=0, sticky="W")
        self.end_date_spin.grid(row=2, column=1)

        self.epsg_label = ttk.Label(self, text="ESPG number for final projection", style="BW.TLabel")
        self.epsg_entry = ttk.Entry(self, style="BW.TEntry")
        self.epsg_label.grid(row=3, column=0, sticky="W")
        self.epsg_entry.grid(row=3, column=1)

        self.s1_label = ttk.Label(self, text="Path to preprocessed sentinel-1 image", style="BW.TLabel")
        self.s1_entry = ttk.Entry(self, style="BW.TEntry")
        self.s1_label.grid(row=4, column=0, sticky="W")
        self.s1_entry.grid(row=4, column=1)

        self.user_label = ttk.Label(self, text="Scihub username:", style="BW.TLabel")
        self.user_entry = ttk.Entry(self, style="BW.TEntry")
        self.user_label.grid(row=5, column=0, sticky="W")
        self.user_entry.grid(row=5, column=1)

        self.passwd_label = ttk.Label(self, text="Scihub password", style="BW.TLabel")
        self.passwd_entry = ttk.Entry(self, style="BW.TEntry")
        self.passwd_label.grid(row=6, column=0, sticky="W")
        self.passwd_entry.grid(row=6, column=1)

        self.run_btn = ttk.Button(self, text="Start analysis")
        self.run_btn.grid(row=7, column=0, columnspan=2)
        # Docs assure me I shouldn't need to do this, how mysterious.
        self.pack()

    def parse_input_to_config(self):
        """Creates configuration file then calls build_cocoa_map with it"""
        # Feels hacky.



if __name__=="__main__":
    interface = CocoaInterface()

