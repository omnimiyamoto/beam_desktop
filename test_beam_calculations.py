import unittest

import numpy as np

from beam_calculations import (
    Beam,
    PointLoad,
    UniformLoad,
    calculate_deflection,
    calculate_forces,
    recommend_section,
    sections_for,
)


class BeamCalculationTests(unittest.TestCase):
    def test_simply_supported_full_uniform_load(self):
        beam = Beam(20, "simple", uniform_loads=(UniformLoad(1, 0, 20),))
        forces = calculate_forces(beam)
        section = sections_for("steel")[2]
        deflection = calculate_deflection(forces, section)
        moment_index = int(np.argmax(np.abs(forces.moment_kip_ft)))
        deflection_index = int(np.argmax(np.abs(deflection)))

        self.assertAlmostEqual(forces.reactions.left_kips, 10)
        self.assertAlmostEqual(forces.reactions.right_kips, 10)
        self.assertAlmostEqual(np.max(np.abs(forces.shear_kips)), 10)
        self.assertAlmostEqual(forces.moment_kip_ft[moment_index], 50, places=2)
        self.assertAlmostEqual(forces.x_ft[moment_index], 10, places=2)
        self.assertAlmostEqual(50 * 12 / section.sx_in3, 17.96, places=2)
        self.assertAlmostEqual(abs(deflection[deflection_index]), 0.609, places=3)
        self.assertAlmostEqual(forces.x_ft[deflection_index], 10, places=2)
        self.assertAlmostEqual(20 * 12 / 360, 0.667, places=3)
        recommendation = recommend_section(forces, "steel", 360)
        self.assertIsNotNone(recommendation)
        if recommendation is not None:
            self.assertEqual(recommendation.name, "W14x22")

    def test_simply_supported_point_load(self):
        beam = Beam(20, "simple", point_loads=(PointLoad(10, 5),))
        forces = calculate_forces(beam)
        section = sections_for("steel")[2]
        deflection = calculate_deflection(forces, section)
        index = int(np.argmax(np.abs(deflection)))

        self.assertAlmostEqual(forces.reactions.left_kips, 7.5)
        self.assertAlmostEqual(forces.reactions.right_kips, 2.5)
        self.assertAlmostEqual(np.max(np.abs(forces.moment_kip_ft)), 37.5)
        self.assertAlmostEqual(37.5 * 12 / section.sx_in3, 13.47, places=2)
        self.assertAlmostEqual(abs(deflection[index]), 0.340, places=3)
        self.assertAlmostEqual(forces.x_ft[index], 8.82, places=2)

    def test_cantilever_free_end_point_load(self):
        beam = Beam(10, "cantilever", point_loads=(PointLoad(5, 10),))
        forces = calculate_forces(beam)
        deflection = calculate_deflection(forces, sections_for("steel")[2])

        self.assertAlmostEqual(forces.reactions.left_kips, 5)
        self.assertAlmostEqual(forces.reactions.fixed_moment_kip_ft, 50)
        self.assertAlmostEqual(np.max(np.abs(deflection)), 0.487, places=3)
        self.assertGreater(np.max(np.abs(deflection)), 10 * 12 / 360)

    def test_lumber_uniform_load(self):
        beam = Beam(12, "simple", uniform_loads=(UniformLoad(0.05, 0, 12),))
        forces = calculate_forces(beam)
        section = sections_for("lumber")[1]
        deflection = calculate_deflection(forces, section)

        self.assertAlmostEqual(np.max(np.abs(forces.moment_kip_ft)), 0.9, places=2)
        self.assertAlmostEqual(0.9 * 12 / section.sx_in3, 0.505, places=3)
        self.assertAlmostEqual(np.max(np.abs(deflection)), 0.147, places=3)
        recommendation = recommend_section(forces, "lumber", 360)
        self.assertIsNotNone(recommendation)
        if recommendation is not None:
            self.assertEqual(recommendation.name, "2x8")

    def test_partial_uniform_load_and_zero_shear_location(self):
        beam = Beam(30, "simple", uniform_loads=(UniformLoad(2, 0, 15),))
        forces = calculate_forces(beam)
        moment_index = int(np.argmax(np.abs(forces.moment_kip_ft)))
        section = sections_for("steel")[5]
        deflection = calculate_deflection(forces, section)
        deflection_index = int(np.argmax(np.abs(deflection)))
        recommendation = recommend_section(forces, "steel", 360)

        self.assertAlmostEqual(forces.reactions.left_kips, 22.5)
        self.assertAlmostEqual(forces.reactions.right_kips, 7.5)
        self.assertAlmostEqual(forces.x_ft[moment_index], 11.25, places=2)
        self.assertAlmostEqual(forces.moment_kip_ft[moment_index], 126.5625, places=3)
        self.assertAlmostEqual(
            abs(forces.moment_kip_ft[moment_index]) * 12 / section.sx_in3,
            26.37,
            places=2,
        )
        self.assertAlmostEqual(abs(deflection[deflection_index]), 1.242, places=3)
        self.assertAlmostEqual(forces.x_ft[deflection_index], 13.79, places=2)
        self.assertGreater(abs(deflection[deflection_index]), 30 * 12 / 360)
        self.assertIsNotNone(recommendation)
        if recommendation is not None:
            self.assertEqual(recommendation.name, "W21x44")

    def test_loads_beyond_span_are_clipped(self):
        beam = Beam(
            20,
            "simple",
            point_loads=(PointLoad(2, 25),),
            uniform_loads=(UniformLoad(1, 15, 30),),
        )
        forces = calculate_forces(beam)

        self.assertAlmostEqual(forces.reactions.left_kips, 0.625)
        self.assertAlmostEqual(forces.reactions.right_kips, 6.375)
        self.assertTrue(np.all(forces.x_ft <= 20))


if __name__ == "__main__":
    unittest.main()
