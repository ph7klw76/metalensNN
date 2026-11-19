import numpy as np
from fiber_metalens_nn.geometry import baselines, epsilon, epsilon_rz, feasible, load_config, project
from fiber_metalens_nn.metrics import full_width


def test_binary_supported_geometry_and_common_baselines():
    config = load_config()
    for heights in baselines(7.8,config).values():
        assert not feasible(heights,config)
        eps,polymer,pec = epsilon_rz(heights,[0.1,1.0,13.0],[-0.05,0.0,0.05,3.0],config)
        assert set(np.unique(eps.real)).issubset({1.0,1.45**2,1.52**2})
        assert np.all(polymer[:,1] >= polymer[:,2])
        assert pec[2,0] and not pec[0,0]
    assert "height grid" in feasible(np.full(32,1.023),config)


def test_width_definitions_and_complex_epsilon_sign():
    x = np.linspace(-3,3,6001)
    intensity = np.exp(-2*x*x/0.7**2)
    assert abs(full_width(x,intensity,np.exp(-2))-1.4) < 0.002
    assert abs(full_width(x,intensity,0.5)-0.7*np.sqrt(2*np.log(2))) < 0.002
    assert np.isclose(epsilon({"n":1.5,"kappa":0.02}).imag,0.06)
