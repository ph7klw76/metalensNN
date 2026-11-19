import json
import numpy as np
from fiber_metalens_nn.geometry import load_config, profile_polygons, edges
from fiber_metalens_nn.maxwell_batch import MaxwellBatch


def test_profile_outline_has_exact_area_and_no_flat_internal_vertices():
    c = load_config()
    h = np.full(32,1.25)
    polygons = profile_polygons(h,c)
    assert len(polygons) == 1 and len(polygons[0]) == 4
    for h in [h,np.array([0,.5,1.,1.,0,2.,.1,0]*4)]:
        area = 0
        for points in profile_polygons(h,c):
            p = np.asarray(points)
            area += .5*abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))
        assert np.isclose(area,np.dot(h,np.diff(edges(c))),rtol=1e-12)


def test_batch_forwards_actual_config_to_worker(tmp_path,monkeypatch):
    c = load_config()
    c['maxwell']['profile_prism'] = True
    batch = MaxwellBatch(c,tmp_path)
    def run(args,**kwargs):
        config = json.loads(open(args[args.index('--config')+1]).read())
        assert config == c
        output = args[args.index('--output')+1]
        with open(output,'w') as stream:
            json.dump({'pass':True},stream)
        return type('Completed',(),{'returncode':0})()
    monkeypatch.setattr('fiber_metalens_nn.maxwell_batch.subprocess.run',run)
    assert batch.job()['pass']


def test_profile_polygon_matches_annular_solid():
    from matplotlib.path import Path
    c = load_config()
    h = np.array([0,.5,1.,1.,0,2.,.1,0]*4)
    polygons = [Path(p) for p in profile_polygons(h,c)]
    r = (edges(c)[1:]+edges(c)[:-1])/2
    z = np.arange(.025,2.5,.05)
    rr,zz = np.meshgrid(r,z,indexing='ij')
    points = np.stack([rr.ravel(),zz.ravel()],axis=-1)
    inside = np.any([p.contains_points(points) for p in polygons],axis=0).reshape(rr.shape)
    assert np.array_equal(inside,zz<h[:,None])
