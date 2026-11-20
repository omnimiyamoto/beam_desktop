"""Tkinter desktop interface for the Beam Designer."""

from __future__ import annotations

import json
import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

import beam_calculations as calculations

BACKGROUND = "#f4f7fb"
CARD = "#ffffff"
INK = "#172033"
MUTED = "#64748b"
BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"


@dataclass(frozen=True)
class SavedDesign:
    beam: calculations.Beam
    material: calculations.MaterialType
    section: str
    deflection_limit: int


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, style="App.TFrame")
        self.canvas = tk.Canvas(
            self, background=BACKGROUND, highlightthickness=0, borderwidth=0
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.content = ttk.Frame(self.canvas, style="App.TFrame", padding=12)
        self._window = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._scroll)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _scroll(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window: tk.Toplevel | None = None
        self._bind_tree(widget)

    def _bind_tree(self, widget: tk.Widget) -> None:
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        for child in widget.winfo_children():
            self._bind_tree(child)

    def show(self, _event: tk.Event) -> None:
        if self.window is not None:
            return
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 12
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.window,
            text=self.text,
            justify="left",
            background=INK,
            foreground="white",
            padx=9,
            pady=6,
            wraplength=310,
        ).pack()

    def hide(self, _event: tk.Event) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


class BeamDesigner:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Beam Designer")
        self.root.geometry("1480x960")
        self.root.minsize(1120, 760)
        self.root.configure(background=BACKGROUND)
        self._build_menu()
        self._configure_styles()

        self.support_display_var = tk.StringVar(value="Simply supported")
        self.span_var = tk.StringVar(value="20")
        self.span_scale_var = tk.DoubleVar(value=20)
        self.material_display_var = tk.StringVar(value="Steel W-shapes")
        self.section_var = tk.StringVar(value="W12x26")
        self.limit_var = tk.StringVar(value="360")
        self.status_var = tk.StringVar(value="Ready")
        self.cursor_var = tk.StringVar(value="Hover over a diagram to inspect values.")
        self.point_loads: list[calculations.PointLoad] = []
        self.uniform_loads = [calculations.UniformLoad(1.0, 0.0, 20.0)]
        self.point_rows: list[tuple[tk.StringVar, tk.StringVar]] = []
        self.uniform_rows: list[
            tuple[tk.StringVar, tk.StringVar, tk.StringVar]
        ] = []
        self._syncing_span = False
        self._ready = False
        self._drag_index: int | None = None
        self._drag_preview = None
        self._force_result: calculations.ForceAnalysis | None = None
        self._deflection = np.array([], dtype=float)
        self._recommendation: calculations.Section | None = None

        self._build_layout()
        self._build_inputs()
        self._build_plot()
        self._ready = True
        self._on_inputs_changed()

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        self.file_menu = tk.Menu(menu, tearoff=False)
        self.file_menu.add_command(label="Save", command=self._save_design)
        self.file_menu.add_command(label="Open", command=self._open_design)
        menu.add_cascade(label="File", menu=self.file_menu)
        self.root.configure(menu=menu)

    def _save_design(self) -> None:
        try:
            beam = self._collect_beam()
            material = self._material_code()
            section = next(
                section
                for section in calculations.sections_for(material)
                if section.name == self.section_var.get()
            )
            divisor = int(self.limit_var.get())
            if divisor not in (240, 360):
                raise ValueError("Deflection limit must be L/240 or L/360.")
        except (ValueError, StopIteration) as exc:
            messagebox.showerror("Save failed", str(exc) or "Choose a listed section.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Beam Design",
            defaultextension=".json",
            filetypes=(("Beam design files", "*.json"), ("All files", "*.*")),
        )
        if not file_path:
            return

        design = {
            "format_version": 1,
            "support": beam.support,
            "span_ft": beam.span_ft,
            "material": material,
            "section": section.name,
            "deflection_limit": divisor,
            "point_loads": [
                {
                    "magnitude_kips": load.magnitude_kips,
                    "position_ft": load.position_ft,
                }
                for load in beam.point_loads
            ],
            "uniform_loads": [
                {
                    "magnitude_kip_per_ft": load.magnitude_kip_per_ft,
                    "start_ft": load.start_ft,
                    "end_ft": load.end_ft,
                }
                for load in beam.uniform_loads
            ],
        }
        try:
            with open(file_path, "w", encoding="utf-8") as design_file:
                json.dump(design, design_file, indent=2)
                design_file.write("\n")
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save the design: {exc}")
            self.status_var.set("Could not save design.")
            return
        self.status_var.set(f"Saved design to {file_path}")

    def _open_design(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Open Beam Design",
            filetypes=(("Beam design files", "*.json"), ("All files", "*.*")),
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as design_file:
                design = self._parse_design(json.load(design_file))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            messagebox.showerror("Open failed", f"Could not open the design: {exc}")
            self.status_var.set("Could not open design.")
            return

        self._apply_design(design)
        self.status_var.set(f"Opened design from {file_path}")

    @staticmethod
    def _parse_design(payload: object) -> SavedDesign:
        if (
            not isinstance(payload, dict)
            or type(payload.get("format_version")) is not int
            or payload.get("format_version") != 1
        ):
            raise ValueError("Unsupported or invalid beam design file.")
        support = payload.get("support")
        if not isinstance(support, str) or support not in ("simple", "cantilever"):
            raise ValueError("Design support must be simple or cantilever.")
        support_code: calculations.SupportType = (
            "simple" if support == "simple" else "cantilever"
        )

        def number(value: object, label: str) -> float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a number.")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"{label} must be finite.")
            return result

        span = number(payload.get("span_ft"), "Span")
        material = payload.get("material")
        if not isinstance(material, str) or material not in calculations.SECTIONS:
            raise ValueError("Design material is not supported.")
        material_code: calculations.MaterialType = (
            "steel" if material == "steel" else "lumber"
        )
        section_name = payload.get("section")
        if not isinstance(section_name, str) or section_name not in {
            section.name for section in calculations.sections_for(material_code)
        }:
            raise ValueError("Design section is not listed for its material.")

        divisor = payload.get("deflection_limit")
        if (
            isinstance(divisor, bool)
            or not isinstance(divisor, int)
            or divisor not in (240, 360)
        ):
            raise ValueError("Deflection limit must be 240 or 360.")

        point_data = payload.get("point_loads")
        uniform_data = payload.get("uniform_loads")
        if not isinstance(point_data, list) or not isinstance(uniform_data, list):
            raise ValueError("Design loads must be lists.")

        points = []
        for item in point_data:
            if not isinstance(item, dict):
                raise ValueError("Each point load must be an object.")
            points.append(
                calculations.PointLoad(
                    number(item.get("magnitude_kips"), "Point load magnitude"),
                    number(item.get("position_ft"), "Point load position"),
                )
            )
        uniforms = []
        for item in uniform_data:
            if not isinstance(item, dict):
                raise ValueError("Each uniform load must be an object.")
            uniforms.append(
                calculations.UniformLoad(
                    number(
                        item.get("magnitude_kip_per_ft"),
                        "Uniform load magnitude",
                    ),
                    number(item.get("start_ft"), "Uniform load start"),
                    number(item.get("end_ft"), "Uniform load end"),
                )
            )

        if any(load.position_ft > 40 for load in points) or any(
            load.start_ft > 40 or load.end_ft > 40 for load in uniforms
        ):
            raise ValueError("Load positions must be between 0 and 40 ft.")
        beam = calculations.Beam(span, support_code, tuple(points), tuple(uniforms))
        calculations.calculate_forces(beam)
        return SavedDesign(beam, material_code, section_name, divisor)

    def _apply_design(self, design: SavedDesign) -> None:
        beam = design.beam
        was_ready = self._ready
        self._ready = False
        try:
            self.support_display_var.set(
                "Simply supported" if beam.support == "simple" else "Cantilever"
            )
            self.span_var.set(f"{beam.span_ft:g}")
            self.span_scale_var.set(beam.span_ft)
            self.material_display_var.set(
                "Steel W-shapes"
                if design.material == "steel"
                else "Douglas Fir-Larch No. 2"
            )
            self._refresh_section_options()
            self.section_var.set(design.section)
            self.limit_var.set(str(design.deflection_limit))
            self.point_loads = list(beam.point_loads)
            self.uniform_loads = list(beam.uniform_loads)
            self._rebuild_load_rows()
        finally:
            self._ready = was_ready
        self._on_inputs_changed()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Card.TLabel", background=CARD, foreground=INK)
        style.configure("Card.Muted.TLabel", background=CARD, foreground=MUTED)
        style.configure(
            "Card.TLabelframe",
            background=CARD,
            bordercolor="#e1e7f0",
            relief="solid",
            padding=10,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=CARD,
            foreground=INK,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Title.TLabel",
            background=BACKGROUND,
            foreground=INK,
            font=("Segoe UI", 17, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background=BACKGROUND,
            foreground=INK,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED)
        style.configure("App.TLabel", background=BACKGROUND, foreground=INK)
        style.configure(
            "Accent.TButton",
            background=BLUE,
            foreground="white",
            padding=(10, 7),
            borderwidth=0,
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")],
        )
        style.configure("Good.Horizontal.TProgressbar", background=GREEN)
        style.configure("Warn.Horizontal.TProgressbar", background=AMBER)
        style.configure("Bad.Horizontal.TProgressbar", background=RED)
        style.configure("Good.TLabel", background="#dcfce7", foreground="#166534")
        style.configure("Bad.TLabel", background="#fee2e2", foreground="#991b1b")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=360)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.input_panel = ScrollableFrame(self.root)
        self.input_panel.grid(
            row=0, column=0, sticky="nsew", padx=(12, 6), pady=12
        )
        self.input_panel.configure(width=370)
        self.input_panel.grid_propagate(False)

        self.main_panel = ttk.Frame(self.root, style="App.TFrame", padding=(6, 12, 12, 12))
        self.main_panel.grid(row=0, column=1, sticky="nsew")
        self.main_panel.columnconfigure(0, weight=1)
        self.main_panel.rowconfigure(1, weight=1)

        header = ttk.Frame(self.main_panel, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(header, text="Beam Designer", style="Title.TLabel").pack(
            side="left"
        )
        ttk.Label(
            header,
            text="Structural analysis and section checks",
            style="App.TLabel",
        ).pack(side="right", pady=(7, 0))

        plot_card = ttk.Frame(
            self.main_panel,
            style="Card.TFrame",
            padding=8,
        )
        plot_card.grid(row=1, column=0, sticky="nsew")
        plot_card.columnconfigure(0, weight=1)
        plot_card.rowconfigure(0, weight=1)
        self.plot_card = plot_card

        footer = ttk.Frame(self.main_panel, style="App.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(footer, textvariable=self.cursor_var, style="App.TLabel").pack(
            side="left"
        )
        ttk.Label(footer, textvariable=self.status_var, style="App.TLabel").pack(
            side="right"
        )

    def _build_inputs(self) -> None:
        content = self.input_panel.content
        content.columnconfigure(0, weight=1)
        ttk.Label(content, text="Beam Setup", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        support_card = self._input_card(content, "Support and span")
        ttk.Label(support_card, text="Support type", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        support = ttk.Combobox(
            support_card,
            textvariable=self.support_display_var,
            values=("Simply supported", "Cantilever"),
            state="readonly",
        )
        support.grid(row=1, column=0, sticky="ew")
        support.bind("<<ComboboxSelected>>", self._on_inputs_changed)
        self.support_box = support

        ttk.Label(support_card, text="Span (ft)", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(10, 4)
        )
        span_row = ttk.Frame(support_card, style="Card.TFrame")
        span_row.grid(row=3, column=0, sticky="ew")
        span_row.columnconfigure(0, weight=1)
        self.span_scale = ttk.Scale(
            span_row,
            from_=4,
            to=40,
            variable=self.span_scale_var,
            command=self._on_span_slider,
        )
        self.span_scale.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.span_entry = ttk.Entry(
            span_row, textvariable=self.span_var, width=8, justify="right"
        )
        self.span_entry.grid(row=0, column=1, sticky="e")
        self.span_var.trace_add("write", self._on_span_entry)

        point_card = self._input_card(content, "Point loads")
        self.point_list = ttk.Frame(point_card, style="Card.TFrame")
        self.point_list.grid(row=0, column=0, sticky="ew")
        self.point_list.columnconfigure(0, weight=1)
        ttk.Button(
            point_card,
            text="+ Add point load",
            command=self._add_point_load,
        ).grid(row=1, column=0, sticky="ew", pady=(7, 0))

        uniform_card = self._input_card(content, "Uniform loads")
        self.uniform_list = ttk.Frame(uniform_card, style="Card.TFrame")
        self.uniform_list.grid(row=0, column=0, sticky="ew")
        self.uniform_list.columnconfigure(0, weight=1)
        ttk.Button(
            uniform_card,
            text="+ Add uniform load",
            command=self._add_uniform_load,
        ).grid(row=1, column=0, sticky="ew", pady=(7, 0))

        section_card = self._input_card(content, "Material and section")
        ttk.Label(section_card, text="Material", style="Muted.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        material_box = ttk.Combobox(
            section_card,
            textvariable=self.material_display_var,
            values=("Steel W-shapes", "Douglas Fir-Larch No. 2"),
            state="readonly",
        )
        material_box.grid(row=1, column=0, sticky="ew")
        material_box.set("Steel W-shapes")
        material_box.bind("<<ComboboxSelected>>", self._on_material_changed)

        ttk.Label(section_card, text="Section", style="Muted.TLabel").grid(
            row=2, column=0, sticky="w", pady=(9, 4)
        )
        self.section_box = ttk.Combobox(
            section_card, textvariable=self.section_var, state="readonly"
        )
        self.section_box.grid(row=3, column=0, sticky="ew")
        self._refresh_section_options()
        self.section_box.bind("<<ComboboxSelected>>", self._on_inputs_changed)

        ttk.Label(section_card, text="Deflection limit", style="Muted.TLabel").grid(
            row=4, column=0, sticky="w", pady=(9, 4)
        )
        limit_box = ttk.Combobox(
            section_card,
            textvariable=self.limit_var,
            values=("360", "240"),
            state="readonly",
        )
        limit_box.grid(row=5, column=0, sticky="ew")
        limit_box.set("360")
        limit_box.bind("<<ComboboxSelected>>", self._on_inputs_changed)

        self._build_check_cards(content)
        self._build_recommendation(content)
        ttk.Label(
            content,
            text="Loads are downward-positive. Self-weight is ignored.",
            style="App.TLabel",
            wraplength=320,
        ).grid(row=10, column=0, sticky="w", pady=(6, 4))

        self._rebuild_load_rows()

    def _input_card(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        row = len(parent.winfo_children())
        card = ttk.LabelFrame(parent, text=title, style="Card.TLabelframe")
        card.grid(row=row, column=0, sticky="ew", pady=(0, 9))
        card.columnconfigure(0, weight=1)
        return card

    def _build_check_cards(self, parent: ttk.Frame) -> None:
        card = self._input_card(parent, "Design checks")
        self.check_widgets: dict[str, dict[str, tk.Widget | tk.StringVar]] = {}
        self._create_check(card, "bending", "Bending", "— ksi", "— ksi")
        self._create_check(card, "deflection", "Deflection", "— in", "— in")

    def _create_check(
        self, parent: ttk.Frame, key: str, title: str, actual: str, allowable: str
    ) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)
        header = ttk.Frame(row, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text=title, style="Card.TLabel").pack(side="left")
        status_var = tk.StringVar(value="PASS")
        status = ttk.Label(header, textvariable=status_var, style="Good.TLabel", padding=(7, 2))
        status.pack(side="right")
        detail_var = tk.StringVar(value=f"Actual {actual}   Allowable {allowable}")
        detail = ttk.Label(row, textvariable=detail_var, style="Muted.TLabel")
        detail.pack(fill="x", pady=(5, 6))
        ratio = tk.DoubleVar(value=0)
        bar = ttk.Progressbar(
            row,
            variable=ratio,
            maximum=100,
            mode="determinate",
            style="Good.Horizontal.TProgressbar",
        )
        bar.pack(fill="x")
        self.check_widgets[key] = {
            "status": status,
            "status_var": status_var,
            "detail_var": detail_var,
            "ratio": ratio,
            "bar": bar,
        }
        explanation = (
            "Bending stress: |Mmax| × 12 / Sx. PASS when the result is no greater "
            "than the material's allowable bending stress."
            if key == "bending"
            else "Deflection is found by integrating M/EI twice using the support "
            "boundary conditions. PASS when the maximum absolute deflection is "
            "no greater than the selected span limit."
        )
        ToolTip(row, explanation)

    def _build_recommendation(self, parent: ttk.Frame) -> None:
        card = self._input_card(parent, "Section recommendation")
        self.recommendation_var = tk.StringVar(value="Calculating…")
        ttk.Label(
            card,
            textvariable=self.recommendation_var,
            style="App.TLabel",
            wraplength=300,
        ).grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.use_section_button = ttk.Button(
            card,
            text="Use recommended section",
            style="Accent.TButton",
            command=self._use_recommendation,
            state="disabled",
        )
        self.use_section_button.grid(row=1, column=0, sticky="ew")

    def _refresh_section_options(self) -> None:
        sections = calculations.sections_for(self._material_code())
        names = tuple(section.name for section in sections)
        self.section_box.configure(values=names)
        if self.section_var.get() not in names:
            self.section_var.set(names[0])

    def _build_plot(self) -> None:
        self.figure = Figure(figsize=(9, 8), dpi=100, facecolor=CARD)
        grid = self.figure.add_gridspec(
            4, 1, height_ratios=(1.65, 1, 1, 1), hspace=0.16
        )
        self.ax_beam = self.figure.add_subplot(grid[0])
        self.ax_shear = self.figure.add_subplot(grid[1])
        self.ax_moment = self.figure.add_subplot(grid[2])
        self.ax_deflection = self.figure.add_subplot(grid[3])
        self.diagram_axes = (
            self.ax_shear,
            self.ax_moment,
            self.ax_deflection,
        )

        self._setup_diagram_axis(self.ax_shear, "SHEAR", "kips", BLUE)
        self._setup_diagram_axis(self.ax_moment, "MOMENT", "kip-ft", PURPLE)
        self._setup_diagram_axis(self.ax_deflection, "DEFLECTION", "in", ORANGE)
        self.ax_deflection.set_xlabel("Position along beam (ft)", color=MUTED)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_card)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._drag_preview = None
        self._crosshairs = [
            axis.axvline(0, color="#64748b", linestyle="--", linewidth=1, visible=False)
            for axis in self.diagram_axes
        ]
        self._plot_fill: dict[str, object | None] = {
            "shear": None,
            "moment": None,
            "deflection": None,
        }
        self._plot_artists: dict[str, tuple[object, object]] = {}
        for name, axis, color in (
            ("shear", self.ax_shear, BLUE),
            ("moment", self.ax_moment, PURPLE),
            ("deflection", self.ax_deflection, ORANGE),
        ):
            line, = axis.plot([], [], color=color, linewidth=1.8, zorder=3)
            marker, = axis.plot([], [], marker="o", color=color, markersize=4, zorder=4)
            annotation = axis.annotate(
                "",
                xy=(0, 0),
                xytext=(5, 7),
                textcoords="offset points",
                color=color,
                fontsize=8,
                fontweight="bold",
                annotation_clip=False,
            )
            self._plot_artists[name] = (line, marker)
            setattr(self, f"_{name}_annotation", annotation)

        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        self.canvas.mpl_connect("figure_leave_event", self._on_plot_leave)

    def _setup_diagram_axis(
        self, axis, title: str, unit: str, color: str
    ) -> None:
        axis.set_facecolor(CARD)
        axis.set_ylabel(f"{title}\n({unit})", color=color, fontsize=8)
        axis.tick_params(axis="both", labelsize=8, colors=MUTED)
        axis.grid(axis="y", color="#e8edf4", linewidth=0.7)
        axis.axhline(0, color="#cbd5e1", linewidth=0.8, zorder=1)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#e1e7f0")
        axis.spines["bottom"].set_color("#e1e7f0")

    def _rebuild_load_rows(self) -> None:
        for widget in self.point_list.winfo_children():
            widget.destroy()
        for widget in self.uniform_list.winfo_children():
            widget.destroy()
        self.point_rows = []
        self.uniform_rows = []

        if not self.point_loads:
            ttk.Label(
                self.point_list,
                text="No point loads",
                style="Muted.TLabel",
            ).grid(row=0, column=0, sticky="w", pady=2)
        for index, load in enumerate(self.point_loads):
            row = ttk.Frame(self.point_list, style="Card.TFrame", padding=(0, 3))
            row.grid(row=index, column=0, sticky="ew", pady=2)
            row.columnconfigure(0, weight=1)
            row.columnconfigure(1, weight=1)
            magnitude = tk.StringVar(value=f"{load.magnitude_kips:g}")
            position = tk.StringVar(value=f"{load.position_ft:g}")
            self.point_rows.append((magnitude, position))
            self._add_load_entry(row, 0, "Magnitude (kips)", magnitude)
            self._add_load_entry(row, 1, "Position (ft)", position)
            ttk.Button(
                row,
                text="×",
                width=3,
                command=lambda i=index: self._remove_load("point", i),
            ).grid(row=0, column=2, rowspan=2, padx=(5, 0), sticky="ns")
            magnitude.trace_add("write", self._on_inputs_changed)
            position.trace_add("write", self._on_inputs_changed)

        if not self.uniform_loads:
            ttk.Label(
                self.uniform_list,
                text="No uniform loads",
                style="Muted.TLabel",
            ).grid(row=0, column=0, sticky="w", pady=2)
        for index, load in enumerate(self.uniform_loads):
            row = ttk.Frame(self.uniform_list, style="Card.TFrame", padding=(0, 3))
            row.grid(row=index, column=0, sticky="ew", pady=2)
            for column in range(3):
                row.columnconfigure(column, weight=1)
            magnitude = tk.StringVar(value=f"{load.magnitude_kip_per_ft:g}")
            start = tk.StringVar(value=f"{load.start_ft:g}")
            end = tk.StringVar(value=f"{load.end_ft:g}")
            self.uniform_rows.append((magnitude, start, end))
            self._add_load_entry(row, 0, "Load (kips/ft)", magnitude)
            self._add_load_entry(row, 1, "Start (ft)", start)
            self._add_load_entry(row, 2, "End (ft)", end)
            ttk.Button(
                row,
                text="×",
                width=3,
                command=lambda i=index: self._remove_load("uniform", i),
            ).grid(row=0, column=3, rowspan=2, padx=(5, 0), sticky="ns")
            magnitude.trace_add("write", self._on_inputs_changed)
            start.trace_add("write", self._on_inputs_changed)
            end.trace_add("write", self._on_inputs_changed)

    def _add_load_entry(
        self, parent: ttk.Frame, column: int, label: str, variable: tk.StringVar
    ) -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(
            row=0, column=column, sticky="w", padx=(0, 4)
        )
        ttk.Entry(parent, textvariable=variable, width=9).grid(
            row=1, column=column, sticky="ew", padx=(0, 4)
        )

    def _read_load_rows(
        self,
    ) -> tuple[list[calculations.PointLoad], list[calculations.UniformLoad]]:
        points: list[calculations.PointLoad] = []
        uniforms: list[calculations.UniformLoad] = []
        for magnitude_var, position_var in self.point_rows:
            magnitude = self._read_number(magnitude_var.get(), "Point load")
            position = self._read_number(position_var.get(), "Point position")
            if magnitude < 0 or not 0 <= position <= 40:
                raise ValueError("Point loads must be nonnegative and positions 0–40 ft.")
            points.append(calculations.PointLoad(magnitude, position))
        for magnitude_var, start_var, end_var in self.uniform_rows:
            magnitude = self._read_number(magnitude_var.get(), "Uniform load")
            start = self._read_number(start_var.get(), "Uniform load start")
            end = self._read_number(end_var.get(), "Uniform load end")
            if magnitude < 0 or not 0 <= start <= 40 or not 0 <= end <= 40:
                raise ValueError("Uniform loads must be nonnegative and positions 0–40 ft.")
            if end < start:
                raise ValueError("Uniform load end must be at or after its start.")
            uniforms.append(calculations.UniformLoad(magnitude, start, end))
        return points, uniforms

    @staticmethod
    def _read_number(text: str, label: str) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value

    def _collect_beam(self) -> calculations.Beam:
        span = self._read_number(self.span_var.get(), "Span")
        if not 4 <= span <= 40:
            raise ValueError("Span must be between 4 and 40 ft.")
        points, uniforms = self._read_load_rows()
        return calculations.Beam(
            span,
            self._support_code(),
            tuple(points),
            tuple(uniforms),
        )

    def _support_code(self) -> calculations.SupportType:
        return "simple" if self.support_display_var.get() == "Simply supported" else "cantilever"

    def _material_code(self) -> calculations.MaterialType:
        return "steel" if self.material_display_var.get() == "Steel W-shapes" else "lumber"

    def _on_span_entry(self, *_args) -> None:
        if not self._ready or self._syncing_span:
            return
        try:
            value = self._read_number(self.span_var.get(), "Span")
            if not 4 <= value <= 40:
                raise ValueError("Span must be between 4 and 40 ft.")
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        snapped = round(value * 4) / 4
        self._syncing_span = True
        self.span_var.set(f"{snapped:g}")
        self.span_scale_var.set(snapped)
        self._syncing_span = False
        self._on_inputs_changed()

    def _on_span_slider(self, value: str) -> None:
        if not self._ready:
            return
        snapped = round(float(value) * 4) / 4
        self._syncing_span = True
        self.span_var.set(f"{snapped:g}")
        self.span_scale_var.set(snapped)
        self._syncing_span = False
        self._on_inputs_changed()

    def _on_material_changed(self, _event: tk.Event | None = None) -> None:
        self._refresh_section_options()
        self._on_inputs_changed()

    def _on_inputs_changed(self, *_args) -> None:
        if not self._ready:
            return
        try:
            beam = self._collect_beam()
            material = self._material_code()
            section = next(
                section
                for section in calculations.sections_for(material)
                if section.name == self.section_var.get()
            )
            divisor = int(self.limit_var.get())
            forces = calculations.calculate_forces(beam)
            deflection = calculations.calculate_deflection(forces, section)
        except (ValueError, StopIteration) as exc:
            self.status_var.set(str(exc) or "Choose a listed section.")
            return

        self.point_loads = list(beam.point_loads)
        self.uniform_loads = list(beam.uniform_loads)
        self._force_result = forces
        self._deflection = deflection
        self._update_checks(forces, section, deflection, divisor)
        self._update_recommendation(forces, material, divisor)
        self._update_plot(forces, deflection)
        self.status_var.set("Ready")

    def _add_point_load(self) -> None:
        if not self._commit_load_rows():
            return
        span = float(self.span_var.get())
        position = span if self._support_code() == "cantilever" else span / 2
        self.point_loads.append(calculations.PointLoad(1.0, position))
        self._rebuild_load_rows()
        self._on_inputs_changed()

    def _add_uniform_load(self) -> None:
        if not self._commit_load_rows():
            return
        span = float(self.span_var.get())
        self.uniform_loads.append(calculations.UniformLoad(0.1, 0.0, span))
        self._rebuild_load_rows()
        self._on_inputs_changed()

    def _remove_load(self, kind: str, index: int) -> None:
        if not self._commit_load_rows():
            return
        if kind == "point":
            del self.point_loads[index]
        else:
            del self.uniform_loads[index]
        self._rebuild_load_rows()
        self._on_inputs_changed()

    def _commit_load_rows(self) -> bool:
        try:
            self.point_loads, self.uniform_loads = self._read_load_rows()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return False
        return True

    def _update_checks(
        self,
        forces: calculations.ForceAnalysis,
        section: calculations.Section,
        deflection: np.ndarray,
        divisor: int,
    ) -> None:
        max_moment = float(np.max(np.abs(forces.moment_kip_ft)))
        bending = max_moment * 12 / section.sx_in3
        bending_ratio = bending / section.allowable_bending_ksi
        max_deflection = float(np.max(np.abs(deflection)))
        limit = forces.beam.span_ft * 12 / divisor
        self._set_check(
            "bending",
            bending,
            section.allowable_bending_ksi,
            "ksi",
            bending_ratio,
        )
        self._set_check(
            "deflection",
            max_deflection,
            limit,
            "in",
            max_deflection / limit,
        )

    def _set_check(
        self, key: str, actual: float, allowable: float, unit: str, ratio: float
    ) -> None:
        widgets = self.check_widgets[key]
        status_var = widgets["status_var"]
        detail_var = widgets["detail_var"]
        ratio_var = widgets["ratio"]
        bar = widgets["bar"]
        status = widgets["status"]
        assert isinstance(status_var, tk.StringVar)
        assert isinstance(detail_var, tk.StringVar)
        assert isinstance(ratio_var, tk.DoubleVar)
        assert isinstance(bar, ttk.Progressbar)
        assert isinstance(status, ttk.Label)

        passes = ratio <= 1
        status_var.set("PASS" if passes else "FAIL")
        status.configure(style="Good.TLabel" if passes else "Bad.TLabel")
        precision = 3 if unit == "in" else 2
        actual_text = f"{actual:.{precision}f} {unit}"
        allowable_text = f"{allowable:.{precision}f} {unit}"
        detail_var.set(f"Actual {actual_text}   Allowable {allowable_text}")
        ratio_var.set(min(100.0, max(0.0, ratio * 100)))
        if ratio < 0.7:
            bar.configure(style="Good.Horizontal.TProgressbar")
        elif ratio <= 1:
            bar.configure(style="Warn.Horizontal.TProgressbar")
        else:
            bar.configure(style="Bad.Horizontal.TProgressbar")

    def _update_recommendation(
        self,
        forces: calculations.ForceAnalysis,
        material: str,
        divisor: int,
    ) -> None:
        recommendation = calculations.recommend_section(
            forces, material, divisor
        )
        self._recommendation = recommendation
        if recommendation is None:
            self.recommendation_var.set("No listed section passes both checks.")
            self.use_section_button.configure(state="disabled")
        else:
            self.recommendation_var.set(
                f"Lightest passing section: {recommendation.name}"
            )
            self.use_section_button.configure(state="normal")

    def _use_recommendation(self) -> None:
        if self._recommendation is None:
            return
        self.section_var.set(self._recommendation.name)
        self._on_inputs_changed()

    def _update_plot(
        self, forces: calculations.ForceAnalysis, deflection: np.ndarray
    ) -> None:
        self._draw_beam(forces)
        self._draw_diagram(
            "shear",
            forces.plot_x_ft,
            forces.plot_shear_kips,
            forces.x_ft,
            forces.shear_kips,
            "kips",
            BLUE,
        )
        self._draw_diagram(
            "moment",
            forces.x_ft,
            forces.moment_kip_ft,
            forces.x_ft,
            forces.moment_kip_ft,
            "kip-ft",
            PURPLE,
        )
        self._draw_diagram(
            "deflection",
            forces.x_ft,
            deflection,
            forces.x_ft,
            deflection,
            "in",
            ORANGE,
        )
        padding = max(0.4, forces.beam.span_ft * 0.035)
        for axis in (self.ax_beam, *self.diagram_axes):
            axis.set_xlim(-padding, forces.beam.span_ft + padding)
        self.canvas.draw_idle()

    def _draw_beam(self, forces: calculations.ForceAnalysis) -> None:
        axis = self.ax_beam
        axis.clear()
        axis.set_facecolor(CARD)
        axis.set_ylim(-1.25, 1.25)
        axis.set_yticks([])
        axis.set_xticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

        beam = forces.beam
        span = beam.span_ft
        axis.plot([0, span], [0, 0], color="#475569", linewidth=7, solid_capstyle="round")
        if beam.support == "simple":
            axis.add_patch(
                Polygon([(-0.35, -0.1), (0.35, -0.1), (0, -0.48)], color="#94a3b8")
            )
            axis.add_patch(
                Polygon(
                    [(span - 0.35, -0.1), (span + 0.35, -0.1), (span, -0.48)],
                    color="#94a3b8",
                )
            )
            reactions = forces.reactions
            for x, reaction in ((0, reactions.left_kips), (span, reactions.right_kips)):
                axis.annotate(
                    "",
                    xy=(x, -0.52),
                    xytext=(x, -0.94),
                    arrowprops={"arrowstyle": "-|>", "color": GREEN, "lw": 1.8},
                )
                axis.text(
                    x,
                    -1.15,
                    f"R = {reaction:.2f} k",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color=GREEN,
                )
        else:
            axis.add_patch(
                Rectangle((-0.18, -0.52), 0.18, 1.04, color="#94a3b8")
            )
            axis.plot([-0.2, -0.45], [-0.52, -0.7], color="#94a3b8", lw=1.2)
            axis.plot([-0.2, -0.45], [-0.28, -0.46], color="#94a3b8", lw=1.2)
            axis.annotate(
                "",
                xy=(0.24, -0.46),
                xytext=(0.24, -0.98),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": GREEN,
                    "lw": 1.8,
                    "connectionstyle": "arc3,rad=-0.25",
                },
            )
            axis.text(
                0.55,
                -1.1,
                f"R = {forces.reactions.left_kips:.2f} k   "
                f"Mfix = {forces.reactions.fixed_moment_kip_ft:.2f} kip-ft",
                ha="left",
                va="top",
                fontsize=8,
                color=GREEN,
            )

        axis.text(0, 0.25, "0 ft", ha="left", fontsize=8, color=MUTED)
        axis.text(span, 0.25, f"{span:g} ft", ha="right", fontsize=8, color=MUTED)
        points = calculations.clip_point_loads(beam)
        uniforms = calculations.clip_uniform_loads(beam)
        for load in points:
            axis.annotate(
                "",
                xy=(load.position_ft, 0.04),
                xytext=(load.position_ft, 0.83),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.7},
            )
            axis.scatter([load.position_ft], [0.02], s=34, color=BLUE, zorder=5)
            axis.text(
                load.position_ft,
                0.91,
                f"{load.magnitude_kips:g} k @ {load.position_ft:g} ft",
                ha="center",
                fontsize=8,
                color=BLUE,
            )
        for load in uniforms:
            if load.end_ft <= load.start_ft:
                continue
            count = max(2, min(30, math.ceil((load.end_ft - load.start_ft) / 0.6)))
            for x in np.linspace(load.start_ft, load.end_ft, count):
                axis.annotate(
                    "",
                    xy=(x, 0.06),
                    xytext=(x, 0.52),
                    arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 0.9},
                )
            axis.text(
                (load.start_ft + load.end_ft) / 2,
                0.58,
                f"{load.magnitude_kip_per_ft:g} k/ft",
                ha="center",
                fontsize=8,
                color=BLUE,
            )

    def _draw_diagram(
        self,
        name: str,
        plot_x: np.ndarray,
        plot_y: np.ndarray,
        sample_x: np.ndarray,
        sample_y: np.ndarray,
        unit: str,
        color: str,
    ) -> None:
        axis = getattr(self, f"ax_{name}")
        line, marker = self._plot_artists[name]
        line.set_data(plot_x, plot_y)
        if self._plot_fill[name] is not None:
            self._plot_fill[name].remove()  # type: ignore[union-attr]
        self._plot_fill[name] = axis.fill_between(
            plot_x, 0, plot_y, color=color, alpha=0.13, zorder=2
        )
        index = int(np.argmax(np.abs(sample_y)))
        max_x = float(sample_x[index])
        max_y = float(sample_y[index])
        marker.set_data([max_x], [max_y])
        annotation = getattr(self, f"_{name}_annotation")
        annotation.xy = (max_x, max_y)
        annotation.set_text(f"{max_y:.3f} {unit} @ {max_x:.2f} ft")
        max_value = float(np.max(np.abs(sample_y)))
        axis.set_ylim(
            -(max_value * 1.32 if max_value else 1),
            max_value * 1.32 if max_value else 1,
        )

    def _on_plot_motion(self, event) -> None:
        if self._drag_index is not None and event.inaxes is self.ax_beam:
            self._preview_drag(event)
            return
        if event.inaxes not in self.diagram_axes or event.xdata is None:
            return
        if self._force_result is None:
            return
        x = float(np.clip(event.xdata, 0, self._force_result.beam.span_ft))
        values = calculations.response_at(
            self._force_result, self._deflection, x
        )
        for crosshair in self._crosshairs:
            crosshair.set_xdata([x, x])
            crosshair.set_visible(True)
        self.cursor_var.set(
            f"x = {x:.2f} ft   V = {values[0]:.2f} k   "
            f"M = {values[1]:.2f} kip-ft   y = {values[2]:.3f} in"
        )
        self.canvas.draw_idle()

    def _on_plot_leave(self, _event) -> None:
        for crosshair in self._crosshairs:
            crosshair.set_visible(False)
        self.cursor_var.set("Hover over a diagram to inspect values.")
        self.canvas.draw_idle()

    def _on_plot_press(self, event) -> None:
        if event.inaxes is not self.ax_beam or event.xdata is None or event.ydata is None:
            return
        if abs(event.ydata) > 0.3 or self._force_result is None:
            return
        points = calculations.clip_point_loads(self._force_result.beam)
        if not points:
            return
        distances = [abs(load.position_ft - event.xdata) for load in points]
        index = int(np.argmin(distances))
        if distances[index] > max(0.25, self._force_result.beam.span_ft * 0.015):
            return
        self._drag_index = index
        self._drag_preview = self.ax_beam.axvline(
            points[index].position_ft, color=BLUE, linestyle="--", linewidth=1.2
        )
        self.canvas.draw_idle()

    def _preview_drag(self, event) -> None:
        if event.xdata is None or self._drag_preview is None:
            return
        span = self._force_result.beam.span_ft if self._force_result else 0
        position = min(span, max(0, round(event.xdata * 4) / 4))
        self._drag_preview.set_xdata([position, position])
        self.status_var.set(f"Move point load to {position:.2f} ft")
        self.canvas.draw_idle()

    def _on_plot_release(self, event) -> None:
        if self._drag_index is None:
            return
        span = self._force_result.beam.span_ft if self._force_result else 0
        moved = event.xdata is not None and self._drag_index < len(self.point_loads)
        if moved:
            position = min(span, max(0, round(event.xdata * 4) / 4))
            load = self.point_loads[self._drag_index]
            self.point_loads[self._drag_index] = calculations.PointLoad(
                load.magnitude_kips, position
            )
        if self._drag_preview is not None:
            self._drag_preview.remove()
            self._drag_preview = None
        self._drag_index = None
        if moved:
            self._rebuild_load_rows()
            self._on_inputs_changed()


def main() -> None:
    root = tk.Tk()
    BeamDesigner(root)
    root.mainloop()


if __name__ == "__main__":
    main()
