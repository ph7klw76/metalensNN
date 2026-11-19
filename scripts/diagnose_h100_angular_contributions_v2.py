"""Decompose authoritative saved H100 Q by forward polar-angle bins.

This locates angular-moment contributions, not their unproven physical cause.
It uses the original512/window128 reconstruction and fixed0.98*n*k0 band.
No alternative acceptance, Meep, JAX, training or raw-data modification.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_height_family_h100_sampling_v2 import cartesian,forward_spectrum,kernels,vector_spectrum,local_path,read,sha,safe_json

SAMPLING_REPORT_SHA='7ddf3b6c610733f7d69561fe82d247afb1a31ff1f82e87656c4a8bdae415ac72'
BINS=(0.,5.,10.,20.,45.,90.)


def angular_bins(spectrum,dx,wavelength,diameter,index=1.,bins=BINS):
    bins=tuple(float(v) for v in bins)
    if len(bins)<2 or bins[0]!=0 or bins[-1]!=90 or any(a>=b for a,b in zip(bins,bins[1:])):
        raise ValueError('Increasing polar-angle bins must cover0 through90 degrees')
    count=spectrum.shape[-1];kx,ky,kz,_,valid,k0=kernels(count,dx,wavelength,n=index)
    kt2=kx*kx+ky*ky
    theta=np.degrees(np.arctan2(np.sqrt(kt2),kz))
    x=(np.arange(count)-count/2+.5)*dx;xx,yy=np.meshgrid(x,x,indexing='xy')
    target=vector_spectrum(np.fft.fft2(np.exp(-(xx*xx+yy*yy)/(diameter/2)**2)),dx,wavelength,n=index)
    def power(s):return np.maximum(.5*np.real(s[0]*np.conj(s[4])-s[1]*np.conj(s[3])),0)*valid
    field=power(spectrum);gaussian=power(target)
    total=float(field.sum());target_total=float(gaussian.sum())
    if not np.isfinite(field).all() or not np.isfinite(gaussian).all() or total<=0 or target_total<=0:
        raise ValueError('Finite positive forward retained-band power required')
    target_moment=float((gaussian*kt2).sum()/target_total)
    field_moment=float((field*kt2).sum()/total)
    if target_moment<=0 or not np.isfinite(target_moment):raise ValueError('Positive finite target moment required')
    rows=[]
    for i,(low,high) in enumerate(zip(bins,bins[1:])):
        selected=valid & (theta>=low) & ((theta<=high) if i==len(bins)-2 else (theta<high))
        power_bin=float(field[selected].sum());target_power_bin=float(gaussian[selected].sum())
        moment_bin=float((field*kt2)[selected].sum()/total)
        target_bin=float((gaussian*kt2)[selected].sum()/target_total)
        q=moment_bin/target_moment;qt=target_bin/target_moment
        rows.append({'theta_min_deg':low,'theta_max_deg':high,
            'field_forward_power_fraction':power_bin/total,'target_forward_power_fraction':target_power_bin/target_total,
            'field_additive_moment_per_um2':moment_bin,'target_additive_moment_per_um2':target_bin,
            'field_additive_Q':q,'target_additive_Q':qt,'additive_Q_excess_over_target':q-qt,
            'fraction_of_field_total_moment':moment_bin/field_moment if field_moment>0 else None})
    Q=field_moment/target_moment
    return {'definition':'Q_bin = sum_bin(A*k_perp^2)/(sum_all(A)*target_moment); additive across bins',
        'power_fraction_normalization':'Reconstructed forward retained-band power, not incident fiber power',
        'target_diameter_um':diameter,'theta_retained_max_deg':float(np.degrees(np.arcsin(.98))),
        'retained_band':'k_perp<0.98*n*k0; last45–90 bin contains only retained45–78.52deg content',
        'Q_unbinned':Q,'field_moment_per_um2':field_moment,'target_moment_per_um2':target_moment,'bins':rows,
        'sum_field_power_fractions':sum(row['field_forward_power_fraction'] for row in rows),
        'sum_target_power_fractions':sum(row['target_forward_power_fraction'] for row in rows),
        'sum_field_additive_Q':sum(row['field_additive_Q'] for row in rows),
        'sum_target_additive_Q':sum(row['target_additive_Q'] for row in rows),
        'sum_additive_Q_excess':sum(row['additive_Q_excess_over_target'] for row in rows)}


def diagnose(folder,expected_sampling_sha=SAMPLING_REPORT_SHA):
    folder=local_path(folder).resolve();source=folder/'diagnostics/H100_sampling_v1/diagnostic.json'
    if sha(source)!=expected_sampling_sha:raise ValueError('Previously verified sampling report changed')
    original=read(source);protected={str(source):sha(source)}
    original_hashes={str(local_path(path).resolve()):expected for path,expected in original['input_sha256'].items()}
    for path,expected in original_hashes.items():
        actual=sha(path)
        if actual!=expected:raise ValueError('Verified source/raw payload changed: '+path)
        protected[str(local_path(path).resolve())]=actual
    config=read(folder/'config.json');D=original['declared_target_diameter_um']
    prior=local_path(read(folder/'prior_stage.json')['folder']).resolve()
    entries={};ledger_hashes={}
    for stage in (prior,folder):
        raw_bytes=(stage/'ledger.json').read_bytes();ledger_hashes[str(stage/'ledger.json')]=hashlib.sha256(raw_bytes).hexdigest()
        for entry in json.loads(raw_bytes.decode('utf-8-sig'))['entries']:
            if entry['id'] in (27,47):entries[entry['id']]=(stage,entry)
    cases=[]
    for identity in (27,47):
        stage,entry=entries[identity];rp=(stage/entry['record_file']).resolve()
        if entry['status']!='completed' or sha(rp)!=entry['record_sha256'] or original_hashes.get(str(rp))!=sha(rp):
            raise ValueError('Completed raw record differs from verified source report')
        raw=read(rp);dx,field=cartesian(raw,512,128.,config)
        spectrum=forward_spectrum(field,dx,config['wavelength_um'],n=config['materials']['ambient']['n'])
        result=angular_bins(spectrum,dx,config['wavelength_um'],D,config['materials']['ambient']['n'])
        saved=next(case for case in original['cases'] if case['id']==identity)['saved_authoritative']['angular_second_moment_ratio']
        result.update({'id':identity,'resolution':raw['settings']['resolution'],'saved_authoritative_Q':saved,
            'binned_minus_saved_Q':result['sum_field_additive_Q']-saved})
        cases.append(result)
    protected[str(Path(__file__).resolve())]=sha(Path(__file__))
    unchanged=all(sha(path)==value for path,value in protected.items())
    if not unchanged:raise RuntimeError('Immutable source/fields changed while diagnosing angular bins')
    return safe_json({'status':'H100_ANGULAR_MOMENT_LOCATION_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),
        'sampling_report':str(source),'sampling_report_sha256':expected_sampling_sha,'sampling':{'count':512,'window_um':128.,'dx_um':.25},
        'same_design_sha256':original['same_design_sha256'],'cases':cases,'input_sha256':protected,
        'ledger_snapshot_sha256':ledger_hashes,'all_immutable_inputs_unchanged':unchanged,
        'new_Maxwell_solves':0,'JAX_or_training_invoked':False,'authoritative_result_reclassified':False,
        'limitations':['Bins identify where the saved angular moment is allocated, not the physical cause of those components.',
            'Q weights power by squared transverse wavevector, so weak large-angle power can contribute disproportionately.',
            'Target contributions sum to1; field contributions sum to the original Q; their differences sum toQ-1.',
            'These are forward retained-band quantities;45–90deg includes no measured78.52–90deg contribution beyond the fixed cutoff.',
            'No thresholds or acceptance results change; no full-angle M2, temporal/Maxwell convergence or experimental claim.']})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--folder',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expected-sampling-sha256',default=SAMPLING_REPORT_SHA)
    args=parser.parse_args();folder=local_path(args.folder).resolve();output=local_path(args.output).resolve()
    if output.exists():raise FileExistsError('Choose a fresh angular diagnostic version')
    if not output.is_relative_to(folder/'diagnostics') or output==folder/'diagnostics':raise ValueError('Output must be a new stage2/diagnostics/version')
    result=diagnose(folder,args.expected_sampling_sha256);output.mkdir(parents=True,exist_ok=False)
    with (output/'diagnostic.json').open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['H100 ANGULAR-MOMENT CONTRIBUTIONS: LOCATION, NOT PHYSICAL CAUSE','Authoritative512/window128 sampling; no acceptance changes.','',
        'mesh angle(deg) forward_power(%) additive_Q target_additive_Q excess_Q']
    for case in result['cases']:
        for row in case['bins']:
            lines.append(f"{case['resolution']:4g} {row['theta_min_deg']:2g}–{row['theta_max_deg']:<2g} "
                f"{100*row['field_forward_power_fraction']:14.8f} {row['field_additive_Q']:.10f} {row['target_additive_Q']:.10f} {row['additive_Q_excess_over_target']:+.10f}")
        lines.append(f"SUM mesh{case['resolution']:g}: power={100*case['sum_field_power_fractions']:.10f}%; Q={case['sum_field_additive_Q']:.12f}; "
                     f"savedQ={case['saved_authoritative_Q']:.12f}; difference={case['binned_minus_saved_Q']:.3g}")
    lines.extend(['',*result['limitations']])
    with (output/'REPORT.txt').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'immutable_inputs_unchanged':result['all_immutable_inputs_unchanged'],
        'sum_Q':[{'resolution':case['resolution'],'Q':case['sum_field_additive_Q'],'difference':case['binned_minus_saved_Q']} for case in result['cases']]},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
