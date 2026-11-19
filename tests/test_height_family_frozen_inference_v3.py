"""Pure decoder-replay guards. No JAX, checkpoint, optimizer or solver use."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE=Path(__file__).resolve().parents[1]/'scripts/verify_height_family_frozen_inference_v3.py'
spec=importlib.util.spec_from_file_location('frozen_inference_pure_tests',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_extracts_only_original_decoder_definitions():
    source='''
def child(folder,stage):
    raise RuntimeError("must not execute child")
    class Model:
        pass
    def phenotype(parameters,H,bundle,method):
        return H
    launch_meep()
'''
    cls,fn,segments=module.extract_decoder_nodes(source)
    namespace={}
    exec(compile(cls,'fixture','exec'),namespace)
    exec(compile(fn,'fixture','exec'),namespace)
    assert namespace['phenotype'](None,75,None,None)==75
    assert 'child' not in namespace
    assert set(segments)=={'Model','phenotype'}


def test_duplicate_or_missing_definitions_are_rejected():
    with pytest.raises(ValueError):module.extract_decoder_nodes('def child(a,b):\n pass\n')
    with pytest.raises(ValueError):module.extract_decoder_nodes('def other(a,b):\n pass\n')


def test_exact_f32_profile_hash_and_height_match():
    expected=np.asarray([0,.05,3],np.float32)
    digest=hashlib.sha256(expected.tobytes()).hexdigest()
    result=module.profile_comparison(expected,expected.tolist(),digest)
    assert result['exact_float32_profile_match']
    assert result['profile_byte_hash_match']
    assert result['maximum_height_difference_um']==0


def test_single_ulp_difference_is_not_relaxed():
    frozen=np.asarray([.05,1],np.float32);actual=frozen.copy()
    actual[0]=np.nextafter(actual[0],np.float32(1))
    result=module.profile_comparison(actual,frozen,hashlib.sha256(frozen.tobytes()).hexdigest())
    assert not result['exact_float32_profile_match']
    assert not result['profile_byte_hash_match']
    assert result['changed_ring_indices_zero_based']==[0]


def test_wrong_hash_and_nonfinite_shape_failures():
    actual=np.asarray([0,3],np.float32)
    assert not module.profile_comparison(actual,actual,'0'*64)['profile_byte_hash_match']
    with pytest.raises(ValueError):module.profile_comparison(actual,[0],'x')
    with pytest.raises(ValueError):module.profile_comparison([0,np.nan],[0,3],'x')


def test_H75_is_interpolation_not_new_fit():
    result=module.interpolation_evidence(75,[40,50,60,70,80,90,100])
    assert result['lower_height_um']==70 and result['upper_height_um']==80
    assert result['lower_weight']==result['upper_weight']==.5
    assert result['no_height_specific_reoptimization']
    assert module.interpolation_evidence(40,[40,50,60])['kind']=='training_table_row'
    with pytest.raises(ValueError):module.interpolation_evidence(75,[40,40,100])
    with pytest.raises(ValueError):module.interpolation_evidence(75,[40,50,60])


def test_hash_guard_catches_actual_byte_change(tmp_path):
    path=tmp_path/'fixture';path.write_bytes(b'original')
    expected=hashlib.sha256(b'original').hexdigest()
    assert module.require_hash(path,expected)==expected
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):module.require_hash(path,expected)
