import numpy as np
from fiber_metalens_nn.collimation_package import cylinder_mesh,mesh_audit,export_stl,caustic_fit


def test_stepped_pedestal_mesh_is_connected_watertight_and_correct_volume(tmp_path):
    heights=np.array([1.,0.,3.,2.,.05,0.])
    vertices,faces=cylinder_mesh(heights,35,24)
    audit=mesh_audit(vertices,faces,heights,35,24)
    assert audit["passed"]
    np.testing.assert_allclose(audit["dimensions_um"],[48,48,38],atol=1e-10)
    path=tmp_path/"pedestal.stl"
    export_stl(path,vertices,faces)
    assert path.stat().st_size==84+50*len(faces)


def test_flat_pedestal_simplifies_without_internal_seams():
    vertices,faces=cylinder_mesh(np.zeros(24),35,24)
    assert len(vertices)==258
    assert mesh_audit(vertices,faces,np.zeros(24),35,24)["passed"]


def test_second_moment_caustic_recovers_known_waist_and_M2():
    wavelength=1.55
    waist=15
    zr=np.pi*waist**2/wavelength
    planes=[]
    for z in (0,25,50,75,100):
        diameter=2*waist*np.sqrt(1+((z-20)/zr)**2)
        planes.append(dict(z_um=z,D4sigma_x_um=diameter,D4sigma_y_um=diameter,edge_power_fraction=0))
    fit=caustic_fit(dict(planes=planes),wavelength)
    assert fit["valid"]
    assert abs(fit["axes"]["x"]["M2"]-1)<1e-8
    assert abs(fit["axes"]["x"]["waist_z_relative_to_output_um"]-20)<1e-8
