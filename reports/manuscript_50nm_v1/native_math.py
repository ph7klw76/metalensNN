"""Small explicit OMML authoring path for the manuscript equations."""
import re
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
def text(value):
    node=OxmlElement('m:r');pr=OxmlElement('m:rPr');style=OxmlElement('m:sty');style.set(qn('m:val'),'p');pr.append(style);node.append(pr)
    wp=OxmlElement('w:rPr');fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),'Cambria Math');fonts.set(qn('w:hAnsi'),'Cambria Math');wp.append(fonts)
    size=OxmlElement('w:sz');size.set(qn('w:val'),'20');wp.append(size);node.append(wp)
    txt=OxmlElement('m:t');txt.set(qn('xml:space'),'preserve');txt.text=str(value);node.append(txt);return node
def container(tag,children):
    node=OxmlElement(tag)
    for child in children:node.append(text(child) if isinstance(child,str) else child)
    return node
def seq(*nodes):return list(nodes)
def sub(base,index):return container('m:sSub',[container('m:e',[base]),container('m:sub',[index])])
def sup(base,power):return container('m:sSup',[container('m:e',[base]),container('m:sup',[power])])
def frac(numerator,denominator):
    return container('m:f',[container('m:num',numerator if isinstance(numerator,list) else [numerator]),container('m:den',denominator if isinstance(denominator,list) else [denominator])])
def root(children):
    pr=OxmlElement('m:radPr');hide=OxmlElement('m:degHide');hide.set(qn('m:val'),'1');pr.append(hide)
    return container('m:rad',[pr,OxmlElement('m:deg'),container('m:e',children if isinstance(children,list) else [children])])
def math(nodes,display=False):
    body=container('m:oMath',nodes)
    if not display:return body
    pr=OxmlElement('m:oMathParaPr');jc=OxmlElement('m:jc');jc.set(qn('m:val'),'center');pr.append(jc)
    return container('m:oMathPara',[pr,body])
def equation(value):
    if value.startswith('E(r'):
        return math(['E(r, 0) = ',sub('E','0'),' exp(−',frac(sup('r','2'),sup(sub('w','f'),'2')),') ,     ',sub('w','f'),' = ',frac('MFD','2'),' = 5.2 µm.    (1)'],True)
    if value.startswith('h_j'):
        return math([sub('h','j'),'(H) = 3 µm × sigmoid{logit[clip(',frac([sub('s','j'),'(H)'],'3 µm'),', 0.001, 0.999)] + ',sub('u','θ,j'),'(ξ)}.    (2)'],True)
    if value.startswith('L ='):
        return math(['L = −η + 0.1',sub('L','D'),' + 0.5',sup('C','2'),' + 0.25',sup('[max(Q − 1, 0)]','2'),' + 0.2',sub('L','s')],True)
    if value.lstrip().startswith('+ 2'):
        return math(['+ 2 × ',sup('10','−4'),sub('L','r'),' + 10',sup('[max(0.9 − T, 0)]','2'),' + ',sub('L','invalid'),'.    (3)'],True)
    if value.startswith('h₅₀'):
        return math([sub('h','50,j'),' = clip{0.05 µm × ',sub('round','even'),'[',frac(sub('h','j'),'0.05 µm'),'], 0, 3 µm}.    (4)'],True)
    if value.startswith('η ='):
        return math(['η = min(',sub('T','native'),', ',sub('T','forward'),') × ',sub('O','G'),'.    (5)'],True)
    if value.startswith('n_j'):
        return math([sub('n','j'),' = rint[',frac(sub('h','50,j'),'0.05 µm'),'],    ',sub('m','j'),' = ⌊',frac([sub('n','j'),' + 5'],'10'),'⌋,    ',sub('h','500,j'),' = 0.5 µm × ',sub('m','j'),'.    (6)'],True)
    if value.startswith('Δz₅₀'):
        return math([sub('Δz','50'),' = 0.05 µm = 0.00005 mm,    ',sub('Δz','500'),' = 0.5 µm = 0.0005 mm.    (7)'],True)
    raise ValueError('No explicit native equation mapping: '+value)

PATTERN=re.compile(r'⟦[A-Z_]+⟧|(?<![A-Za-z_])([A-Za-z])_(\([^)]*\)|[A-Za-zθ⊥]+(?:,j)?)([²³]?)')
def inline_nodes(token,match):
    if token=='⟦TARGET_DIAMETER⟧':return [sub('D','t'),' = 2',root(['2',sub('⟨r²⟩','I')])]
    if token=='⟦SIDE_RADIUS⟧':return [sub('w','t'),'(z)',root(frac('ln(100)','2'))]
    if token=='⟦RAYLEIGH⟧':return [sub('z','R'),' = ',frac(['π',sub('n','air'),sup(sub('D','t'),'2')],['4',sub('λ','0')])]
    if token=='⟦CURVATURE⟧':return [frac(sub('z','R'),'R')]
    base,index,power=match.groups();index=index.strip('()');node=sub(base,index)
    if power:node=sup(node,{'²':'2','³':'3'}[power])
    return [node]
def add_inline(paragraph,value,add_text):
    pos=0
    for match in PATTERN.finditer(value):
        if match.start()>pos:add_text(value[pos:match.start()])
        paragraph._p.append(math(inline_nodes(match.group(0),match)))
        pos=match.end()
    if pos<len(value):add_text(value[pos:])
