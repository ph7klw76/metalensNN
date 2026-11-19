"""Windows desktop interface for the frozen height-conditioned fiber metalens."""

import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk


PROJECT = Path(__file__).resolve().parents[1]
BACKGROUND = '#ebf1f6'
INK = '#14283a'
MUTED = '#486073'
WHITE = '#ffffff'
LINE = '#d4e0e8'
FINE = '#176283'
COARSE = '#97501a'
ACCENT = '#174b78'


def cylinder_height(value):
    try:
        height = float(value)
    except ValueError as exc:
        raise ValueError('Enter a cylinder height, such as 67.5.') from exc
    if not math.isfinite(height) or not 40 <= height <= 100 or height * 2 != round(height * 2):
        raise ValueError('Use 40–100 µm in 0.5 µm increments to preserve uniform SLC layers.')
    return height


def run_inference(height):
    portable_python = PROJECT / 'runtime' / 'python.exe'
    if portable_python.is_file():
        command = [str(portable_python)]
    elif (PROJECT / 'runtime').is_dir():
        raise RuntimeError('The portable runtime is missing. Restore the complete distribution folder.')
    else:
        python = Path(sys.executable)
        if python.name.lower() == 'pythonw.exe':
            python = python.with_name('python.exe')
        command = [str(python)]
    command.extend(['-B', '-X', 'utf8', 'scripts/fiber_metalens_desktop_inference.py',
                    '--height-um', str(height), '--backend', 'cpu'])
    try:
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, encoding='utf-8',
                                errors='replace', timeout=240,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('The optical prediction exceeded its four-minute time limit.') from exc
    if result.returncode in (124, 137):
        raise RuntimeError('The optical prediction stopped before returning both profiles.')
    if result.returncode:
        error = result.stderr.strip().splitlines()
        raise RuntimeError(error[-1] if error else 'The inference worker returned an error.')
    return validate_result(json.loads(result.stdout), height)


def validate_result(result, height):
    if (result['height_um'] != height or result['cylinder_diameter_um'] != 50
            or result['radial_edges_um'] != [i / 2 for i in range(51)]):
        raise ValueError('The inference result does not match the requested geometry.')
    for key in ('design_50nm', 'design_500nm'):
        design = result[key]
        values = design['heights_um']
        if (len(values) != 50 or any(not math.isfinite(z) or not 0 <= z <= 3.000002 for z in values)
                or not 0 <= design['useful_modal_efficiency'] <= 1):
            raise ValueError(f'Invalid geometry or efficiency returned for {key}.')
    fine = result['design_50nm']['heights_um']
    coarse = result['design_500nm']['heights_um']
    fine_grid = [round(z * 20) for z in fine]
    if (any(abs(z - tick / 20) > 2e-6 for z, tick in zip(fine, fine_grid))
            or coarse != [(tick + 5) // 10 / 2 for tick in fine_grid]):
        raise ValueError('The 500 nm profile does not match the project’s rounding rule.')
    return result


class ProfilePlot(tk.Canvas):
    def __init__(self, parent, color):
        super().__init__(parent, background=WHITE, highlightthickness=0, height=220, width=305)
        self.color = color
        self.values = None
        self.bind('<Configure>', self.draw)

    def update_profile(self, values):
        self.values = values
        self.draw()

    def draw(self, _event=None):
        self.delete('all')
        width, height = self.winfo_width(), self.winfo_height()
        if width < 120 or height < 100:
            return
        left, right, top, bottom = 48, width - 23, 26, height - 45
        x = lambda radius: left + radius / 25 * (right - left)
        y = lambda relief: bottom - relief / 3.2 * (bottom - top)
        if self.values is not None:
            points = []
            for index, relief in enumerate(self.values):
                points.extend((x(index / 2), y(relief), x((index + 1) / 2), y(relief)))
            self.create_polygon(left, bottom, *points, right, bottom,
                                fill='#e3f0f5' if self.color == FINE else '#f6ebdf', outline='')
        for tick in (0, 1, 2, 3):
            self.create_line(left, y(tick), right, y(tick), fill=LINE)
            self.create_text(left - 10, y(tick), text=str(tick), anchor='e', fill=MUTED)
        for tick in (0, 5, 10, 15, 20, 25):
            self.create_line(x(tick), bottom, x(tick), bottom + 5, fill=MUTED)
            self.create_text(x(tick), bottom + 16, text=str(tick), fill=MUTED)
        self.create_line(left, top, left, bottom, right, bottom, fill=MUTED)
        self.create_text(left, 10, text='Relief height (µm)', anchor='w', fill=MUTED)
        self.create_text((left + right) / 2, height - 9, text='Radius (µm)', fill=MUTED)
        if self.values is None:
            self.create_text((left + right) / 2, (top + bottom) / 2,
                             text='Generate a design to plot the profile', fill=MUTED)
        else:
            self.create_line(*points, fill=self.color, width=2)


class DesignWindow:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.events = Queue()
        self.busy = False
        self.exported_path = None
        root.title('Fiber Metalens Designer')
        root.geometry('1010x715')
        root.minsize(800, 635)
        root.configure(background=BACKGROUND)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol('WM_DELETE_WINDOW', self.close)

        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('Accent.TButton', background=ACCENT, foreground=WHITE,
                        font=('Segoe UI', 10, 'bold'), padding=(15, 7), borderwidth=0)
        style.map('Accent.TButton', background=[('disabled', '#9bacbb'), ('active', '#206392')],
                  foreground=[('disabled', '#f2f6fa')])
        style.configure('Action.TButton', font=('Segoe UI', 10), padding=(12, 7))
        style.configure('Height.TSpinbox', padding=(6, 5))

        header = tk.Frame(root, background=INK, padx=23, pady=16)
        header.grid(row=0, column=0, sticky='ew')
        self.label(header, 'Fiber Metalens Designer', size=22, bold=True, color=WHITE, bg=INK).pack(anchor='w')
        self.label(header, '1550 nm  •  50 µm cylinder diameter  •  Polymer index 1.5263  •  Fiber MFD 10.4 µm',
                   color='#c4d7e7', bg=INK).pack(anchor='w', pady=(3, 0))

        controls = tk.Frame(root, bg=WHITE, padx=16, pady=12, highlightbackground=LINE, highlightthickness=1)
        controls.grid(row=1, column=0, sticky='ew', padx=18, pady=(14, 10))
        self.label(controls, 'Cylinder height', bold=True).grid(row=0, column=0, sticky='w', padx=(0, 12))
        self.height = tk.StringVar(value='75')
        self.input = ttk.Spinbox(controls, from_=40, to=100, increment=.5, width=9,
                                 textvariable=self.height, style='Height.TSpinbox', font=('Segoe UI', 13))
        self.input.grid(row=0, column=1, sticky='w')
        self.label(controls, 'µm', color=MUTED).grid(row=0, column=2, padx=(7, 15))
        self.generate_button = ttk.Button(controls, text='Generate design', style='Accent.TButton', command=self.generate)
        self.generate_button.grid(row=0, column=3, sticky='w')
        self.label(controls, '40–100 µm  •  0.5 µm increments', color=MUTED).grid(row=0, column=4, sticky='w', padx=14)
        self.height.trace_add('write', self.height_changed)
        self.input.bind('<Return>', lambda _event: self.generate())

        cards = tk.Frame(root, bg=BACKGROUND)
        cards.grid(row=2, column=0, sticky='nsew', padx=18)
        cards.rowconfigure(0, weight=1)
        cards.columnconfigure((0, 1), weight=1, uniform='profiles')
        self.cards = [self.card(cards, 0, '50 nm relief', 'Fine design from the trained weights', FINE),
                      self.card(cards, 1, '500 nm relief', 'Fabrication profile used for SLC export', COARSE)]

        footer = tk.Frame(root, bg=WHITE, padx=16, pady=12, highlightbackground=LINE, highlightthickness=1)
        footer.grid(row=3, column=0, sticky='ew', padx=18, pady=12)
        footer.columnconfigure(0, weight=1)
        self.label(footer, 'Export cylinder and relief', bold=True, size=12).grid(row=0, column=0, sticky='w')
        self.export_button = ttk.Button(footer, text='Save 500 nm SLC…', style='Accent.TButton',
                                        command=self.export, state='disabled')
        self.export_button.grid(row=0, column=1, sticky='e', padx=(12, 0))
        self.status = tk.StringVar(value='Enter a height and select Generate design.')
        status = tk.Label(footer, textvariable=self.status, bg=WHITE, fg=MUTED, justify='left',
                          anchor='w', font=('Segoe UI', 10))
        status.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(5, 0))
        footer.bind('<Configure>', lambda event: status.configure(wraplength=max(650, event.width - 36)))
        self.open_button = ttk.Button(footer, text='Open export folder', style='Action.TButton',
                                     command=self.open_export, state='disabled')
        self.open_button.grid(row=0, column=2, sticky='e', padx=(8, 0))
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=117)
        self.progress.grid(row=1, column=2, sticky='e', pady=(5, 0))
        self.progress.grid_remove()
        self.input.focus_set()
        self.root.after(70, self.receive_events)

    @staticmethod
    def label(parent, text='', *, size=10, bold=False, color=INK, bg=WHITE):
        return tk.Label(parent, text=text, bg=bg, fg=color, anchor='w',
                        font=('Segoe UI', size, 'bold' if bold else 'normal'))

    def card(self, parent, column, title, description, color):
        card = tk.Frame(parent, bg=WHITE, padx=17, pady=12, highlightbackground=LINE, highlightthickness=1)
        card.grid(row=0, column=column, sticky='nsew', padx=((0, 5) if column == 0 else (5, 0)))
        card.columnconfigure(0, weight=1)
        card.rowconfigure(4, weight=1)
        self.label(card, title, size=16, bold=True).grid(row=0, column=0, sticky='w')
        self.label(card, description, color=MUTED).grid(row=1, column=0, sticky='w', pady=(1, 7))
        efficiency = self.label(card, '—', size=27, bold=True, color=color)
        efficiency.grid(row=2, column=0, sticky='w')
        self.label(card, 'Predicted useful modal efficiency (optical proxy)', color=MUTED).grid(row=3, column=0, sticky='w')
        plot = ProfilePlot(card, color)
        plot.grid(row=4, column=0, sticky='nsew', pady=(5, 2))
        geometry = self.label(card, '50 radial bands  •  0.5 µm wide', color=MUTED)
        geometry.grid(row=5, column=0, sticky='w')
        return efficiency, plot, geometry

    def discard_result(self):
        self.result = None
        self.exported_path = None
        self.export_button.configure(state='disabled')
        self.open_button.configure(state='disabled')
        for efficiency, plot, geometry in self.cards:
            efficiency.configure(text='—')
            plot.update_profile(None)
            geometry.configure(text='50 radial bands  •  0.5 µm wide')

    def height_changed(self, *_):
        if not self.busy:
            self.discard_result()
            self.status.set('Select Generate design for the current height.')

    def working(self, value):
        self.busy = value
        state = 'disabled' if value else 'normal'
        self.input.configure(state=state)
        self.generate_button.configure(state=state)
        self.export_button.configure(state='normal' if not value and self.result else 'disabled')
        self.open_button.configure(state='normal' if not value and self.exported_path else 'disabled')
        if value:
            self.progress.grid()
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.grid_remove()

    def start_task(self, kind, work):
        self.working(True)

        def run():
            try:
                self.events.put((kind, work(), None))
            except Exception as exc:
                self.events.put((kind, None, exc))

        threading.Thread(target=run, daemon=True).start()

    def generate(self):
        if self.busy:
            return
        self.discard_result()
        try:
            height = cylinder_height(self.height.get())
        except ValueError as exc:
            self.status.set(str(exc))
            self.input.focus_set()
            return
        self.status.set(f'Evaluating the trained model and both relief profiles for H = {height:g} µm…')
        self.start_task('design', lambda: run_inference(height))

    def display_design(self, result):
        if cylinder_height(self.height.get()) != result['height_um']:
            raise ValueError('The height changed. Generate a design for the current height.')
        for key, (efficiency, plot, geometry) in zip(('design_50nm', 'design_500nm'), self.cards):
            design = result[key]
            efficiency.configure(text=f"{design['useful_modal_efficiency']:.2%}")
            plot.update_profile(design['heights_um'])
            geometry.configure(text=f"Max. relief {design['max_relief_um']:.2f} µm   •   "
                                    f"Assembly height {design['assembly_top_um']:.2f} µm")
        self.result = result
        self.status.set(f"H = {result['height_um']:g} µm. Profiles ready. SLC export uses 0.50 µm layers.")

    def export(self):
        if self.busy or not self.result:
            return
        result = self.result
        height = result['height_um']
        if cylinder_height(self.height.get()) != height:
            self.height_changed()
            return
        export_folder = PROJECT / 'exports'
        name = filedialog.asksaveasfilename(parent=self.root, title='Save verified 500 nm assembly',
            initialdir=export_folder if export_folder.is_dir() else PROJECT, initialfile=f'Metalens_H{height:g}um_500nm.slc',
            defaultextension='.slc', filetypes=[('SLC contours', '*.slc')], confirmoverwrite=False)
        if not name:
            return
        path = Path(name)
        if path.exists():
            self.status.set('Choose a new filename to preserve the existing SLC.')
            return
        self.status.set('Writing the SLC and independently checking dimensions, layers and contours…')
        self.start_task('export', lambda: export_design(path, result))

    def display_export(self, output):
        if output['audit_status'] != 'PASS':
            raise ValueError('The exported geometry did not pass its independent audit.')
        self.exported_path = Path(output['slc_path'])
        self.status.set(f"Geometry verified · {output['layer_count']} layers × 0.50 µm · "
                        f"{output['total_height_um']:g} µm total\n{self.exported_path}")

    def receive_events(self):
        try:
            kind, output, error = self.events.get_nowait()
        except Empty:
            pass
        else:
            try:
                if error is not None:
                    raise error
                if kind == 'design':
                    self.display_design(output)
                else:
                    self.display_export(output)
            except Exception as exc:
                if kind == 'design':
                    self.discard_result()
                self.status.set(f"{'Could not generate a design' if kind == 'design' else 'Could not export'}: {exc}")
            finally:
                self.working(False)
        self.root.after(70, self.receive_events)

    def open_export(self):
        if self.exported_path is not None:
            try:
                os.startfile(self.exported_path.parent)
            except (AttributeError, OSError) as exc:
                self.status.set(f'Could not open the export folder: {exc}')

    def close(self):
        if self.busy:
            self.status.set('The current operation is still running. Close the window when it finishes.')
        else:
            self.root.destroy()


def export_design(path, result):
    from fiber_metalens_desktop_slc import export_slc

    design = result['design_500nm']
    return export_slc(path, result['height_um'], design['heights_um'], inference_provenance={
        'model': result['provenance'], 'backend': result['backend'],
        'predicted_useful_modal_efficiency': design['useful_modal_efficiency'],
        'fine_profile_sha256': result['design_50nm']['profile_sha256'],
        'export_profile_sha256': design['profile_sha256'],
    })


def main():
    root = tk.Tk()
    DesignWindow(root)
    root.mainloop()


if __name__ == '__main__':
    main()
