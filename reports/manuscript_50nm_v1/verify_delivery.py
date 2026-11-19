"""Bind the final DOCX to source checks and complete rendered-page review."""
from pathlib import Path
from hashlib import sha256
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];QA=HERE/'qa_v4'
DOC=ROOT/'reports/word/07_NN_Fiber_Collimation_50nm_Manuscript_v4.docx'
def digest(path):return sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
assert digest(DOC)=='a2b1a0784c779c6bf0d03687456b9bb4539f989674f0d403bd3394e057d0c0a4'
assert digest(DOC)==read(HERE/'docx_build_manifest.json')['sha256']
for name in ('A_structure_review.json','C_review.json','root_visual_review.json'):
    assert read(QA/name)['status'].startswith('PASS'),name
visual=read(QA/'root_visual_review.json')
assert visual['final_changed_pages_reviewed_at_original_resolution']==[5,7]
assert len(list(QA.glob('page-*.png')))==14
for n in range(1,15):
    proof=visual['page_equivalence_to_reviewed_v3'][str(n)]
    assert digest(QA/f'page-{n}.png')==proof['v4_sha256']
    if n not in (5,7):assert proof['identical']
evidence=[HERE/'docx_build_manifest.json',HERE/'manuscript_content.json',HERE/'native_math.py',HERE/'evidence/manuscript_facts.json',HERE/'evidence/manuscript_claim_review.json',QA/'A_structure_review.json',QA/'C_review.json',QA/'root_visual_review.json',HERE/'qa_v3/B_review.json']
gate={'status':'PASS','deliverable':str(DOC),'docx_sha256':digest(DOC),'bytes':DOC.stat().st_size,
 'pages':14,'main_figures':[1,2,3,4,5],'supplementary_figures':['S1'],'tables':3,
 'display_equations':7,'native_display_math_blocks':8,'native_Word_math':True,
 'numeric_profile_scope':'Original50nm-quantized NN designs; separate500nm geometry explanation',
 'coarse_500nm_optical_performance_claimed':False,'new_optical_solves':0,'new_training_runs':0,
 'visual_review_coverage':visual['review_coverage'],'complete_final_page_png_sha256':{str(n):digest(QA/f'page-{n}.png') for n in range(1,15)},
 'review_evidence_sha256':{str(p.relative_to(HERE)):digest(p) for p in evidence},
 'resolved_findings':['Title paragraph rule removed','All mathematical notation typeset as native Word math','Performance table kept together','Figure3 caption spacing and expression wrap corrected','All body callouts present','Page7 residual word confirmed complete']}
with (HERE/'delivery_gate.json').open('x',encoding='utf-8') as f:f.write(json.dumps(gate,indent=2)+'\n')
print(json.dumps({'status':'PASS','docx':str(DOC),'pages':14,'sha256':digest(DOC)}))
