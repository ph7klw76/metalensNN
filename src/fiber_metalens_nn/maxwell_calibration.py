"""Small uniform-field check of cylindrical sub-aperture flux quadrature."""
import time
from pathlib import Path
import meep as mp
from .trial import write_json


def main():
    start = time.monotonic()
    checks = []
    for resolution in [20,30,40]:
        frequency = 1/.98
        pulse = mp.GaussianSource(frequency,fwidth=.3*frequency,is_integrated=True)
        sim = mp.Simulation(cell_size=mp.Vector3(3,0,6),dimensions=mp.CYLINDRICAL,m=1,
                            resolution=resolution,boundary_layers=[mp.PML(1)],
                            default_material=mp.Medium(index=1.45),force_complex_fields=True,
                            accurate_fields_near_cylorigin=True,Courant=.45,
                            sources=[mp.Source(pulse,component=c,amplitude=a,center=mp.Vector3(1.5,0,-1),size=mp.Vector3(3,0,0))
                                     for c,a in [(mp.Er,1),(mp.Ep,1j)]])
        monitors = [sim.add_flux(frequency,0,1,mp.FluxRegion(center=mp.Vector3(r/2,0,.5),size=mp.Vector3(r,0,0),direction=mp.Z))
                    for r in [.55,1.1]]
        sim.run(until_after_sources=20)
        fluxes = [float(mp.get_fluxes(monitor)[0]) for monitor in monitors]
        ratio = fluxes[0]/fluxes[1]
        checks.append({"resolution":resolution,"radii_um":[.55,1.1],"powers":fluxes,"area_ratio_expected":.25,
                       "measured_ratio":ratio,"relative_error":abs(ratio/.25-1),"pass":abs(ratio/.25-1)<.02})
        sim.reset_meep()
    write_json(Path("artifacts/phase7/collection-quadrature-check.json"),
               {"result_label":"SYNTHETIC_DEMONSTRATION","homogeneous_calibration_not_candidate":True,
                "checks":checks,"pass":all(value["pass"] for value in checks),"wall_s":time.monotonic()-start})


if __name__ == "__main__":
    main()
