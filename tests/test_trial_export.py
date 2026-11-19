import json
import numpy as np
from fiber_metalens_nn.trial import write_json


def test_scientific_report_export(tmp_path):
    path = tmp_path/"report.json"
    write_json(path,{"pass":np.bool_(True),"power":np.float32(.5),"heights":np.array([0,.05])})
    assert json.loads(path.read_text()) == {"pass":True,"power":.5,"heights":[0,.05]}
