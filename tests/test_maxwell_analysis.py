import numpy as np
from fiber_metalens_nn.maxwell_batch import relative_checks
from fiber_metalens_nn.geometry import load_config
from fiber_metalens_nn.maxwell_analysis import analyze_fields


def test_vector_azimuthal_power_reconstruction():
    # Arbitrary complex radial fields, including longitudinal components.
    # The reflected m=-1 sector has E=(Er,-Ep,Ez), H=(-Hr,Hp,-Hz).
    er,ep,ez = 1+.3j,.2+.7j,.1+.1j
    hr,hp,hz = -.1-.6j,.9+.2j,.2-.1j
    phi = np.arange(4096)*2*np.pi/4096
    linear_er,linear_ep = er*np.cos(phi),1j*ep*np.sin(phi)
    linear_hr,linear_hp = 1j*hr*np.sin(phi),hp*np.cos(phi)
    average_linear_flux = np.mean(.5*np.real(linear_er*np.conj(linear_hp)-linear_ep*np.conj(linear_hr)))
    circular_flux = .5*np.real(er*np.conj(hp)-ep*np.conj(hr))
    assert np.isclose(average_linear_flux,.5*circular_flux,rtol=1e-13)
    # A forward circular plane wave becomes uniform x-linear illumination.
    ex = (np.cos(phi)**2+np.sin(phi)**2)
    ey = np.cos(phi)*np.sin(phi)-np.sin(phi)*np.cos(phi)
    assert np.allclose(ex,1) and np.allclose(ey,0)


def test_maxwell_convergence_requires_power_closure_and_interior_focus():
    config = load_config("configs/desktop_trial.json")
    a = {"focal_um":7.8,"eta_R":.3,"focus_on_scan_boundary":False,"power_balance_pass":True}
    b = dict(a,focal_um=7.81,eta_R=.301)
    assert relative_checks(a,b,config)["pass"]
    assert not relative_checks(a,dict(b,power_balance_pass=False),config)["pass"]
    assert not relative_checks(a,dict(b,focus_on_scan_boundary=True),config)["pass"]


def test_near_axis_metrics_do_not_use_zero_point_monitor(tmp_path):
    z = np.arange(4.,12.01,.25)
    r = np.linspace(0,3.3,100)
    field = np.exp(-((z[:,None]-7.8)/.6)**2)*np.exp(-(r[None,:]/.4)**2)
    point = field[:,0]**2
    point[1::2] = 0
    path = tmp_path/"fields.npz"
    np.savez(path,Er=field,Ep=1j*field,Ez=np.zeros_like(field),r_um=r,focus_scan_um=z,axis_intensity=point)
    result = analyze_fields({"fields_file":str(path),"focal_um":8.,"settings":{"resolution":30}})
    assert abs(result["focal_um"]-7.8) < .02
    assert abs(result["widths"]["fwhm_x_um"]-.4*np.sqrt(2*np.log(2))) < .01
    with np.load(path) as data:
        assert np.array_equal(data["axis_point_intensity_raw"],point)
        assert np.all(data["axis_intensity"] > 0)
