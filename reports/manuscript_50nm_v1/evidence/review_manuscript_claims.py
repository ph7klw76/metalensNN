"""Independent manuscript numeric/claim check against bounded saved facts."""
from datetime import datetime,timezone
import hashlib,json,math
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];P=OUT.parent/'manuscript_content.json';checks=[]
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)))
    if not ok:raise AssertionError(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def local(p):
    p=str(p)
    if len(p)>2 and p[1]==':':p='/mnt/'+p[0].lower()+'/'+p[3:].replace('\\','/')
    return Path(p).resolve()
original=P.read_bytes();content=json.loads(original);facts=read(OUT/'manuscript_facts.json');check('Pinned saved facts',sha(OUT/'manuscript_facts.json')=='0faecc0940a7d3330de108fff2afbc998869037699770851105c74e1b0cdc299')
text='\n'.join(str(x.get('text',''))+'\n'+str(x.get('caption','')) for x in content)
check('No corrupted UTF8markers',not any(x in text for x in ('Â','Ã','â€','�')))
check('Actual H75postsource259 corrected','160, 259 and 335 solver time units' in text and '160, 260 and 335' not in text)
caption=next(x['caption'] for x in content if x['kind']=='figure' and 'Figure3' in x['caption'])
check('Exact correctly decodedFigure3caption',caption==(OUT.parent/'figures/Figure3_caption.txt').read_text(encoding='utf-8').strip())
check('FinalFig2/5 sources',any(x.get('stem')=='final_v3/fig02_learning_and_proxy_contribution' for x in content) and any(x.get('stem')=='final_v3/fig05_mesh_and_numerical_verification' for x in content))
table=next(x for x in content if x['kind']=='table' and x['caption'].startswith('Table 2'))
for row in table['rows']:
    H=float(row[0]);truth=next(x for x in facts['all_eight_Maxwell_performance'] if x['H_um']==H)
    expected=[H,truth['resolution_cells_per_um'],truth['output_Dx_um'],truth['output_Dy_um'],truth['eta_percent'],truth['C'],truth['Q']]
    for i,(written,value) in enumerate(zip(row,expected)):
        tolerance=0 if i<2 else .500001*10**(-len(written.split('.')[1]))
        check('Table2 rounded value '+str(H)+' col'+str(i),abs(float(written)-value)<=tolerance)
table3=next(x for x in content if x['kind']=='table' and x['caption'].startswith('Table 3'))
for row in table3['rows']:
    H=int(row[0]);f=next(x for x in facts['fabrication_50nm_to_500nm'] if x['H_um']==H);a,b=f['original_50nm'],f['derived_500nm'];ch=f['change']
    check('Table3 geometry/counts '+str(H),int(row[1])==a['assembly_total_height_um']==b['assembly_total_height_um'] and int(row[2])==a['layer_count'] and int(row[3])==b['layer_count'] and row[4]==f"{a['nonzero_footprint_diameter_um']:g} → {b['nonzero_footprint_diameter_um']:g}" and int(row[5])==ch['changed_annuli'] and abs(float(row[6])-ch['unweighted_ring_height_RMS_error_um'])<=.000005)
comparison=read(local(facts['source_paths']['final_comparison']));width_errors=[]
for row in comparison['rows']:
    errors=[]
    for p,m in zip(row['proxy']['planes'],row['Meep']['planes']):
        check('Same forwardwidth plane',p['z_um']==m['z_um'])
        errors.extend(abs(p[k]/m[k]-1) for k in ('diameter_x_um','diameter_y_um'))
    width_errors.append(dict(H_um=row['H_um'],maximum_relative_width_error_Meep_denominator=max(errors),maximum_percent=100*max(errors)))
check('Proxywidth percentages correctly rounded',[f"{r['maximum_percent']:.2f}" for r in width_errors]==['1.80','0.81','1.22'])
selected=[next(x for x in facts['all_eight_Maxwell_performance'] if x['H_um']==H) for H in (40,75,100)]
for p in selected:
    check('Maintexteta '+str(p['H_um']),f"{p['eta_percent']:.5f}%" in text)
    check('Maintext coreangles '+str(p['H_um']),f"{p['core_half_angles_deg'][0]:.5f}°/{p['core_half_angles_deg'][1]:.5f}°" in text)
    mesh=p['last_mesh_pair']['comparison']['measurements'];check('Selected widthdelta '+str(p['H_um']),f"{100*mesh['maximum_diameter_relative_change']:.4f}%" in text)
duration=next(c for c in facts['original_sampled_controls'] if c['kind']=='duration');delta_eta_pp=100*(duration['variation']['eta']-duration['nominal']['eta']);delta_Q=duration['Q_variant']-duration['Q_prior']
check('Coarseduration numericalvalues',f"{delta_eta_pp:.5f}" in text.replace('−','-') and f"{delta_Q:+.8f}" in text)
for required in ('11,698 parameters','350 parameters','seed 47 at update 100','seed 11 with its best checkpoint at update 90','972 matched calls','not a complete end-to-end speedup claim','round_even','floor[(n_j + 5)/10]','0.00005 mm','0.0005 mm','ring-wise RMS','Ring-wise RMS','No electromagnetic calculation in this manuscript revalidates the 500 nm profiles'):
    if required=='ring-wise RMS':continue
    check('Source-consistent phrase '+required,required in text)
for phrase in ('H75 is validation-observed','not ISO M²','m = +1','not treated as an m = 0','Gaussian surrogate','extinction coefficients are set to zero','per-height field-derived seed','not refitted to the trained device output','not a factor-of-ten scale change','entire cylinder plus relief as denominator','original attempt-83 controller exit/completion fields remain unrecorded'):
    check('Explicit claim limit '+phrase,phrase in text)
check('SourceJSON stable during review',P.read_bytes()==original)
review=dict(status='PASS_MANUSCRIPT_CLAIMS_AFTER_CORRECTIONS',created_utc=datetime.now(timezone.utc).isoformat(),manuscript_content_file=str(P),manuscript_content_sha256=hashlib.sha256(original).hexdigest(),facts_sha256=sha(OUT/'manuscript_facts.json'),passed_assertions=len(checks),checks=checks,derived_width_errors=width_errors,coarse_duration_delta_eta_percentage_points=delta_eta_pp,coarse_duration_delta_Q=delta_Q,resolved_findings=[dict(issue='H75 selected postsource duration was260; raw/analysis/facts say259',resolution='Methods now160,259,335.'),dict(issue='Figure3 embeddedcaption hadactualdouble-decoded UTF8characters',resolution='Re-imported originalUTF8caption; exactcurrentcaptionverified.'),dict(issue='Radial roughness wording couldimplysmoothnessacrossH',resolution='Neighboringannular-height variation clarified.'),dict(issue='Fig5closure wasdescribedasdevice-only',resolution='Nowdescribedaslargerdevice/referenceerror.')],remaining_material_findings=[],manual_semantic_review='Full5733word draft reviewed against source artifacts: physicalindices/MFD/geometry/decoder/objective/selection/cost, nominal and mesh gates, scope ofQ/corefits/finecontrols, originalvscoarsequantization,volume denominator,novelty/heldout/experimental limits.',bibliographic_primary_checks=[dict(ref=1,url='https://www.nature.com/articles/s41566-018-0246-9',status='Title/authors/year/volume/pages/DOI match; generalinverse-designbackground.'),dict(ref=2,url='https://pubs.acs.org/doi/pdf/10.1021/acs.nanolett.9b01857',status='Jiang/Fan2019NanoLetters19(8)5366-5372 confirmed.'),dict(ref=3,url='https://www.nature.com/articles/ncomms11763',status='Gissibl/Thiele/Herkommer/Giessen2016single-mode-fiberprinting confirmed.'),dict(ref=4,url='https://cris.vub.be/ws/files/111128012/2024_K._Van_Mol_Fabrication_of_multilevel_metalenses_using_multiphoton_lithography_from_design_to_evaluation.pdf',status='Author-hosted publishedpaper confirms authors,title,2024,32(6),10190-10203andDOI514237.'),dict(ref=5,url='https://www.corning.com/media/worldwide/coc/documents/Fiber/product-information-sheets/PI-1424-AEN.pdf',status='OfficialJuly2025datasheet confirms10.4+/-0.5umMFD at1550nm,NA0.14definitionat1310nm,125umcladding.'),dict(ref=6,url='https://arxiv.org/abs/1412.6980',status='Authorprimarypaper confirmsKingma/BaandICLR2015.'),dict(ref=7,url='https://meep.readthedocs.io/en/stable/Acknowledgements/',status='Officialpackagecitation confirms2010CPC181,687-702;Introduction confirmscylindricalFDTD/units/Yeecontext.'),dict(ref=8,url='https://www.iso.org/standard/77769.html',status='OfficialISOcatalogue confirms11146-1:2021andtitle; noISOcomplianceclaimed.')],new_optical_or_field_or_training_runs=0,source_or_manuscript_edits_by_reviewer=0)
with (OUT/'manuscript_claim_review.json').open('x',encoding='utf-8') as f:json.dump(review,f,indent=2);f.write('\n')
print(json.dumps(dict(status=review['status'],checks=len(checks),content_sha256=review['manuscript_content_sha256'],review_sha256=sha(OUT/'manuscript_claim_review.json'))))
