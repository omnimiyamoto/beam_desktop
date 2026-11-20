import csv
import os
import tempfile
import tkinter as tk
import unittest
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import patch

import beam_calculations as calculations
from app import BeamDesigner


class BeamDesignerInputTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.root.withdraw()
        self.app = BeamDesigner(self.root)
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.update()
            self.root.destroy()

    @staticmethod
    def _find_button(parent, text):
        for child in parent.winfo_children():
            if isinstance(child, ttk.Button) and child.cget("text") == text:
                return child
            nested = BeamDesignerInputTests._find_button(child, text)
            if nested is not None:
                return nested
        return None

    def test_beam_and_load_controls_update_results(self):
        self.assertAlmostEqual(self.app._force_result.beam.span_ft, 20)
        self.assertAlmostEqual(max(abs(self.app._force_result.moment_kip_ft)), 50)

        self.app.span_var.set("24")
        self.assertAlmostEqual(self.app._force_result.beam.span_ft, 24)
        self.app._on_span_slider("12")
        self.assertEqual(self.app.span_var.get(), "12")
        self.assertAlmostEqual(self.app._force_result.beam.span_ft, 12)

        self.app.support_display_var.set("Cantilever")
        self.app._on_inputs_changed()
        self.assertEqual(self.app._force_result.beam.support, "cantilever")

        add_point = self._find_button(self.app.point_list.master, "+ Add point load")
        self.assertIsNotNone(add_point)
        add_point.invoke()
        self.assertEqual(len(self.app.point_rows), 1)
        self.assertAlmostEqual(float(self.app.point_rows[0][1].get()), 12)
        self.app.point_rows[0][0].set("5")
        self.app.point_rows[0][1].set("12")
        self.assertAlmostEqual(self.app._force_result.reactions.left_kips, 17)
        self.assertAlmostEqual(
            self.app._force_result.reactions.fixed_moment_kip_ft, 132
        )

        self.app._on_plot_press(
            SimpleNamespace(inaxes=self.app.ax_beam, xdata=12.0, ydata=0.0)
        )
        self.app._on_plot_release(
            SimpleNamespace(inaxes=self.app.ax_beam, xdata=7.13, ydata=0.0)
        )
        self.assertEqual(self.app.point_rows[0][1].get(), "7.25")

        self.app._on_plot_motion(
            SimpleNamespace(inaxes=self.app.ax_shear, xdata=3.0)
        )
        self.assertIn("x = 3.00 ft", self.app.cursor_var.get())
        self.assertTrue(all(line.get_visible() for line in self.app._crosshairs))
        self.app._on_plot_leave(SimpleNamespace())
        self.assertTrue(all(not line.get_visible() for line in self.app._crosshairs))

        remove_point = self._find_button(self.app.point_list, "×")
        self.assertIsNotNone(remove_point)
        remove_point.invoke()
        self.assertEqual(len(self.app.point_rows), 0)

        self.app.support_display_var.set("Simply supported")
        self.app._on_inputs_changed()
        add_uniform = self._find_button(
            self.app.uniform_list.master, "+ Add uniform load"
        )
        self.assertIsNotNone(add_uniform)
        add_uniform.invoke()
        self.assertEqual(len(self.app.uniform_rows), 2)
        self.app.uniform_rows[1][0].set("0.5")
        self.app.uniform_rows[1][1].set("0")
        self.app.uniform_rows[1][2].set("6")
        self.assertGreater(
            max(abs(self.app._force_result.moment_kip_ft)), 9
        )

        remove_uniform = self._find_button(self.app.uniform_list, "×")
        self.assertIsNotNone(remove_uniform)
        remove_uniform.invoke()
        self.assertEqual(len(self.app.uniform_rows), 1)

    def test_material_section_and_limit_controls_update_checks(self):
        self.app.material_display_var.set("Douglas Fir-Larch No. 2")
        self.app._on_material_changed()
        self.assertEqual(self.app.section_box.cget("values"), ("2x8", "2x10", "2x12"))
        self.assertEqual(self.app.section_var.get(), "2x8")

        self.app.section_var.set("2x10")
        self.app._on_inputs_changed()
        self.assertIn("Allowable 0.90 ksi", self.app.check_widgets["bending"]["detail_var"].get())

        self.app.limit_var.set("240")
        self.app._on_inputs_changed()
        self.assertIn("Allowable 1.000 in", self.app.check_widgets["deflection"]["detail_var"].get())

        self.app.material_display_var.set("Steel W-shapes")
        self.app._on_material_changed()
        self.assertIn("W8x10", self.app.section_box.cget("values"))

    def test_save_and_open_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = os.path.join(directory, "design.json")
            self.app.support_display_var.set("Cantilever")
            self.app.span_var.set("28")
            self.app.material_display_var.set("Douglas Fir-Larch No. 2")
            self.app._on_material_changed()
            self.app.section_var.set("2x10")
            self.app.limit_var.set("240")
            self.app.point_loads = [calculations.PointLoad(4.5, 7.25)]
            self.app.uniform_loads = [calculations.UniformLoad(0.3, 2, 18)]
            self.app._rebuild_load_rows()
            self.app._on_inputs_changed()

            with (
                patch("app.filedialog.asksaveasfilename", return_value=file_path),
                patch("app.messagebox.showerror") as showerror,
            ):
                self.app.file_menu.invoke(0)
            showerror.assert_not_called()
            self.assertTrue(os.path.isfile(file_path))

            self.app.support_display_var.set("Simply supported")
            self.app.span_var.set("20")
            self.app.material_display_var.set("Steel W-shapes")
            self.app._on_material_changed()
            self.app.section_var.set("W12x26")
            self.app.limit_var.set("360")
            self.app.point_loads = []
            self.app.uniform_loads = []
            self.app._rebuild_load_rows()
            self.app._on_inputs_changed()

            with (
                patch("app.filedialog.askopenfilename", return_value=file_path),
                patch("app.messagebox.showerror") as showerror,
            ):
                self.app.file_menu.invoke(1)
            showerror.assert_not_called()

        self.assertEqual(self.app.support_display_var.get(), "Cantilever")
        self.assertEqual(self.app.span_var.get(), "28")
        self.assertEqual(self.app.material_display_var.get(), "Douglas Fir-Larch No. 2")
        self.assertEqual(self.app.section_var.get(), "2x10")
        self.assertEqual(self.app.limit_var.get(), "240")
        self.assertEqual(
            self.app._collect_beam(),
            calculations.Beam(
                28,
                "cantilever",
                (calculations.PointLoad(4.5, 7.25),),
                (calculations.UniformLoad(0.3, 2, 18),),
            ),
        )

    def test_invalid_open_preserves_current_design(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = os.path.join(directory, "invalid.json")
            with open(file_path, "w", encoding="utf-8") as design_file:
                design_file.write('{"format_version": 1}')

            with (
                patch("app.filedialog.askopenfilename", return_value=file_path),
                patch("app.messagebox.showerror") as showerror,
            ):
                self.app._open_design()

        showerror.assert_called_once()
        self.assertEqual(self.app._collect_beam().span_ft, 20)
        self.assertEqual(self.app.section_var.get(), "W12x26")

    def test_export_results_csv_matches_calculated_values(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = os.path.join(directory, "results.csv")
            with (
                patch("app.filedialog.asksaveasfilename", return_value=file_path),
                patch("app.messagebox.showerror") as showerror,
            ):
                self.app.export_button.invoke()
            showerror.assert_not_called()

            with open(file_path, "r", newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = list(reader)
                self.assertEqual(
                    reader.fieldnames,
                    ["x (ft)", "V (kips)", "M (kip-ft)", "Deflection (in)"],
                )

        self.assertEqual(len(rows), len(self.app._force_result.x_ft))
        midspan = next(row for row in rows if float(row["x (ft)"]) == 10.0)
        self.assertAlmostEqual(float(midspan["V (kips)"]), 0)
        self.assertAlmostEqual(float(midspan["M (kip-ft)"]), 50)
        self.assertAlmostEqual(
            abs(float(midspan["Deflection (in)"])), 0.609, places=3
        )

    def test_calculation_sheet_pdf_exports_without_changing_plot_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            file_path = os.path.join(directory, "calculation_sheet.pdf")
            original_top = self.app.figure.subplotpars.top
            savefig = self.app.figure.savefig

            def savefig_with_header(*args, **kwargs):
                header_text = "\n".join(
                    text.get_text() for text in self.app.figure.texts
                )
                self.assertIn("Support: Simply supported", header_text)
                self.assertIn("Uniform loads: 1 k/ft from 0 to 20 ft", header_text)
                self.assertIn("Bending:", header_text)
                self.assertIn("Deflection:", header_text)
                return savefig(*args, **kwargs)

            with (
                patch("app.filedialog.asksaveasfilename", return_value=file_path),
                patch("app.messagebox.showerror") as showerror,
                patch.object(
                    self.app.figure, "savefig", side_effect=savefig_with_header
                ),
            ):
                self.app.calculation_sheet_button.invoke()
            showerror.assert_not_called()

            with open(file_path, "rb") as pdf_file:
                self.assertEqual(pdf_file.read(5), b"%PDF-")
            self.assertAlmostEqual(self.app.figure.subplotpars.top, original_top)
            self.assertIn("Saved calculation sheet", self.app.status_var.get())


if __name__ == "__main__":
    unittest.main()
