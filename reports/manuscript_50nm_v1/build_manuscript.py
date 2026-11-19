"""Assemble an evidence-linked computational manuscript with editable text."""
from pathlib import Path
import argparse, hashlib, json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import native_math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
F=json.loads((HERE/'evidence/manuscript_facts.json').read_text(encoding='utf-8'))
PERF=F['all_eight_Maxwell_performance'];LEARN=F['learning'];FAB=F['fabrication_50nm_to_500nm']
SELECT={int(r['H_um']):r for r in PERF if r['H_um'] in [40,75,100]}
OUT=ROOT/'reports/word/07_NN_Fiber_Collimation_50nm_Manuscript.docx'
blocks=[]
def p(s):blocks.append(dict(kind='p',text=s))
def h(s,level=1):blocks.append(dict(kind='h',text=s,level=level))
def eq(s):blocks.append(dict(kind='eq',text=s))
def fig(stem,caption,width=6.6):
    if stem in ('fig02_learning_and_proxy_contribution','fig05_mesh_and_numerical_verification'):
        stem='final_v3/'+stem
    blocks.append(dict(kind='figure',stem=stem,caption=caption,width=width))
def table(caption,headers,rows,widths=None):blocks.append(dict(kind='table',caption=caption,headers=headers,rows=rows,widths=widths))

def content():
    blocks.append(dict(kind='title',text='Physics seeded neural inverse design of fiber mounted polymer collimators'))
    blocks.append(dict(kind='subtitle',text='A computational study at 1550 nm with 50 nm relief quantization'))
    h('Abstract')
    p('Fiber-mounted micro-optics require a coordinated treatment of source propagation, finite support thickness, surface relief and fabrication discretization. We investigate a height-conditioned neural inverse-design workflow for polymer collimators operating at a vacuum wavelength of 1550 nm. Each device comprises a 50 µm diameter solid cylinder and a concentric relief with 50 radial annuli, 0.5 µm radial width and a maximum height of 3 µm. A neural network modifies a field-derived phase seed, while a finite-thickness differentiable optical model supplies the training objective. The selected reliefs are quantized vertically in 50 nm increments and checked independently using cylindrical finite-difference time-domain simulations. Across eight assessed heights between 40 and 100 µm, useful modal efficiencies range from 95.288% to 95.559% under the assumed lossless materials and Gaussian source. The 40, 75 and 100 µm designs produce output intensity diameters of approximately 11.6, 14.4 and 16.7 µm, respectively. Their proxy and Maxwell beam-width predictions differ by at most 1.80%, 0.81% and 1.22% at the five declared processing planes. The results demonstrate near-flat output wavefronts and propagation close to predeclared Gaussian references, subject to the measured numerical checks. They do not establish neural-network superiority or a global diffraction optimum. We also explain the separate conversion to 500 nm SLC fabrication layers: the fine relief is requantized, changing individual annular heights by up to 250 nm. Geometry audits of these coarse files do not transfer optical acceptance from the original designs.')
    p('Keywords  Fiber micro-optics; neural inverse design; polymer collimator; finite-difference time-domain; relief quantization; two-photon fabrication')
    h('1 Introduction')
    p('Compact collimation at an optical-fiber output can reduce the size of downstream beam-handling systems. A structure printed directly on the fiber can combine a mechanical standoff with a phase-correcting surface, but its optical response depends on more than the phase profile assigned to the top face. Light diffracts through the support, encounters material interfaces and interacts with the finite cylindrical boundary. The source mode, cylinder height, local relief and output reference plane therefore form one optical problem.')
    p('Computational inverse design provides a way to search such structures using a specified optical objective [1]. Neural approaches can distribute optimization over families of devices or learn reusable design corrections [2]. These capabilities require evidence beyond a low training loss: the generated geometry must retain its performance in an independent electromagnetic calculation, and any claimed benefit over a conventional optimizer must be evaluated with consistent constraints and computational accounting. Direct printing of free-form optics and multilevel lenses on fibers has already been demonstrated [3,4]. The motivation here is consequently the quantitative assessment of a physics-seeded design family and its translation into explicitly audited fabrication geometry.')
    p('We study a fixed source and material system while allowing a different top-relief pattern at each cylinder height. This differs from asking one unchanged pattern to work over a 40–100 µm height range. The optical target is a flat-phase Gaussian with a height-dependent natural footprint, propagated over a 100 µm processing interval. A finite collimated Gaussian continues to diffract; constant beam diameter over that interval is not imposed. The earlier aspiration to obtain an approximately 30 µm output beam is not demonstrated by the present 12–17 µm output diameters.')
    p('The manuscript separates three levels of evidence. The differentiable model is used for design and proxy comparisons; independent Maxwell fields determine the reported optical response of the original 50 nm profiles; and SLC audits determine whether an exported solid matches a specified fabrication profile. No fabricated-device or experimental optical result is reported. Figures 1–5 document the device model, learning behavior, physical profiles, collimation and numerical checks. The later fabrication section explains why a 500 nm layer export is a distinct geometry rather than a new label for the verified 50 nm design.')
    h('2 Methods')
    h('Optical source and connected geometry',2)
    p('The vacuum wavelength is λ₀ = 1.55 µm. The source is a centered, linearly polarized Gaussian amplitude at the fiber facet. The 1550 nm mode-field diameter is 10.4 µm, based on the nominal Corning SMF-28 Ultra specification [5]. The associated datasheet tolerance is ±0.5 µm, but that tolerance is not sampled in the reported optical campaign. The approximately 8 µm core label and 125 µm cladding diameter are not substituted for the emitted mode-field diameter. The nominal fiber NA of 0.14 is retained as a manufacturer descriptor, not imposed as a hard angular cutoff on the Gaussian field.')
    eq('E(r, 0) = E₀ exp(−r²/w_f²),      w_f = MFD/2 = 5.2 µm.                                      (1)')
    p('The input-medium phase index is assumed to be 1.45. The polymer index is 1.5263 at 1550 nm, as specified for this study, and the exterior index is 1. All extinction coefficients are set to zero. The input is a Gaussian surrogate, rather than a measured vector mode or a solved core–cladding eigenmode. The supplied real polymer index does not establish measured absorption, dispersion or post-development shrinkage.')
    p('The facet is z = 0. The cylinder radius is 25 µm and its top reference is z = H. The top surface is represented by 50 annuli with boundaries r_j = 0.5j µm, j = 0,…,50. Each annular height h_j lies between 0 and 3 µm. The complete solid occupies 0 ≤ z ≤ H + h_j for r_j ≤ r < r_(j+1). Cylinder and relief use the same material and are represented by one connected exterior profile, avoiding artificial internal contact faces. Outer annuli with h_j = 0 retain the cylinder below H. The highest possible surface is H + 3 µm; the output monitor is at H + 4 µm and a second native-field monitor is at H + 9 µm.')
    table('Table 1  Optical and geometric inputs', ['Parameter','Value and interpretation'],[
      ['Vacuum wavelength','1.55 µm'],['Input field','Centered x-linear Gaussian; MFD 10.4 µm at z = 0'],
      ['Phase indices','Input 1.45 assumed; polymer 1.5263 supplied; air 1.00'],['Loss model','All material extinction coefficients zero'],
      ['Cylinder','Diameter 50 µm; separate designs at each H'],['Relief discretization','50 annuli; radial width 0.5 µm; height 0–3 µm'],
      ['Original vertical quantization','0.05 µm = 50 nm'],['Output and processing coordinates','Output z = H + 4 µm; downstream distances 0, 25, 50, 75, 100 µm']], [.35, .65])
    fig('figure_1_device_and_workflow','Figure 1. Device configuration and physics-seeded neural inverse-design workflow. The left panel defines the solid cylinder, concentric top relief and output-monitor origin. Its envelope and exaggerated relief are schematic, not simulated intensity data. The right panel separates continuous training through the differentiable model from quantized validation and independent Meep evaluation. Each height receives its own phase seed and top pattern. The network predicts geometry corrections; it does not predict the plotted Maxwell fields.')
    h('Height dependent optical target and neural representation',2)
    p('For each H, the finite-cylinder proxy first computes a complex field at the cylinder top. A phase-derived annular relief supplies the initial seed. The intensity-weighted radial second moment of this independently computed field defines a scalar Gaussian diameter parameter D_t = 2[2⟨r²⟩_I]¹ᐟ². A flat-phase target with this parameter is placed at the output reference and projected onto the retained vector spectrum in air. Its x/y intensity crossings are measured after propagation to each processing plane and frozen before optimization. Thus the target is not refitted to the trained device output.')
    p('The network input is the normalized cylinder height ξ = (H − 70 µm)/(30 µm). Three dense hidden layers, each with 64 tanh units, feed 50 residual logits. The model has 11,698 parameters. The output kernels and biases are initially zero; the hidden layers use their framework defaults. A bounded decoder combines the residual u_θ,j with a per-height field-derived seed s_j. The fraction s_j/3 µm is clipped to [0.001, 0.999] before applying the logit, avoiding singular endpoints.')
    eq('h_j(H) = 3 µm × sigmoid{logit[clip(s_j(H)/(3 µm), 0.001, 0.999)] + u_θ,j(ξ)}.             (2)')
    p('This is a physics-seeded residual generator conditioned only on H. Its use at a new height also requires construction of the corresponding optical seed and target bundle. It has not been trained to accept arbitrary MFD, NA, refractive index, wavelength or beam target. A non-neural comparator uses a 7 × 50 table of residual logits at the training heights, giving 350 parameters, with linear interpolation of residuals at other heights. The comparator uses the same field seeds, bounded decoder, quantization and objective.')
    h('Differentiable model and objective',2)
    p('The design model uses one-way scalar split-step propagation through the finite polymer cylinder and relief, followed by vector E/H continuation in air. The configured transverse grid has 192 samples over 64 µm with padding factor 2, giving an effective 384-sample, 128 µm window and a 0.3333 µm transverse increment. The pedestal step is 0.5 µm and the relief is represented by 24 slices over 3 µm. Approximate interface amplitudes are included, but backward waves and complete vector boundary coupling are not supplied by this model. Those approximations motivate the independent Maxwell check.')
    p('The objective rewards useful modal power and penalizes width error, phase-current curvature, excess retained-band angular moment, excess power outside a target-defined radius, neighboring-height variation and low transmission. Width terms include both axes at all five processing planes. With numerical heights expressed in micrometres, the implemented objective is')
    eq('L = −η + 0.1L_D + 0.5C² + 0.25[max(Q − 1, 0)]² + 0.2L_s')
    eq('       + 2 × 10⁻⁴L_r + 10[max(0.9 − T, 0)]² + L_invalid.                                      (3)')
    p('Here L_D is the mean squared relative deviation from the frozen vector-target diameters; L_r is the mean squared difference of neighboring annular heights; and L_invalid is the fraction of unresolved diameter crossings. L_s is the mean squared excess, above 0.01, of positive axial power outside the target Gaussian 99%-power radius. Its radius is w_t(z)[ln(100)/2]¹ᐟ². The sidelobe metric is therefore a power fraction outside a declared boundary, not a sidelobe peak intensity. C and Q are defined below using the same physical references used in reporting.')
    h('Training selection and exact quantization',2)
    p('Training heights are 40, 50, 60, 70, 80, 90 and 100 µm. Validation heights are 45, 55, 65, 75, 85 and 95 µm. Five heights, 42.5, 57.5, 72.5, 87.5 and 97.5 µm, were initially withheld from checkpoint and seed selection and later evaluated with the proxy. Consequently H40 and H100 are training heights, and H75 is validation-observed. None is described here as an untouched held-out test. Independent Maxwell assessment of the nominal family does not establish generalization to all intermediate heights.')
    p('NN and non-neural arms each use random seeds 11, 29 and 47 and 100 updates. A training height is sampled uniformly at every update. Adam [6] follows global-gradient-norm clipping at 1, with learning rates 0.004 for the NN and 0.025 for the direct table. Continuous relief is used during differentiation. At every tenth update, the decoded heights are quantized to 50 nm and evaluated at the six fixed validation heights. The selection key is lexicographic: summed normalized constraint violations, negative worst validation useful efficiency, and mean validation loss. The selected NN is seed 47 at update 100; the selected direct-table solution is seed 11 with its best checkpoint at update 90.')
    eq('h₅₀,j = clip{0.05 µm × round_even[h_j/(0.05 µm)], 0, 3 µm}.                              (4)')
    p('The original quantizer uses the nearest-even rule at exact half-grid ties. The saved heights retain float32 representation, giving very small deviations from their nominal 50 nm levels. Every arm records 100 training calls, 60 validation calls and two trained-state recovery calls, for 162 calls per arm and 972 matched calls across the six arms. This accounting excludes pilot gradients, per-height seed/target construction and final proxy reporting; it is not a complete end-to-end speedup claim. Replay of selected checkpoint parameters reproduces the frozen NN and direct profiles at H40, H75 and H100 exactly. Training and proxy calculations used JAX on an RTX 4060; Maxwell validation ran independently on the CPU.')
    h('Independent Maxwell calculation and field normalization',2)
    p('The frozen quantized profiles are evaluated with Meep 1.35.0 [7] in cylindrical coordinates. A connected r–z prism represents the complete cylinder and relief. Centered linear polarization is reconstructed using the m = +1 solution and its m = −1 counterpart; it is not treated as an m = 0 field. The Courant factor is 0.45. A Gaussian temporal pulse, centered at 1/λ₀ with frequency width 0.2/λ₀ in the solver units, supplies complex Fourier E/H fields at 1550 nm. The calculation is not reported as continuous-wave excitation or as complete temporal convergence of every angular moment.')
    p('Each device uses a homogeneous reference with matching source, input phase index, discretization, boundaries and monitor settings. The nominal PML thickness is 2 µm, radial padding is 10 µm and the native output-monitor radius is 35 µm. Post-source durations for the selected H40, H75 and H100 records are 160, 260 and 335 solver time units, respectively. Matched native power, reflection subtraction and control-surface closure are retained. The selected fields use 32 cells/µm for H40/H75 and 40 cells/µm for H100; their source profiles remain unchanged during refinement.')
    p('Analysis reconstructs the centered linearly polarized E/H fields on a 512 × 512 Cartesian grid over a 128 µm window. The frozen paraxial Yee power-equivalent correction and matched-field forward decomposition are retained. Angular-spectrum propagation in homogeneous air supplies the downstream planes and is cross-checked against the native monitor at H + 9 µm. Only the fixed propagating band k_⊥ < 0.98n_air k₀ is retained. Native versus reconstructed transmission, aperture fractions, monitor alignment, propagating-filter loss and power closure are checked before optical metrics are accepted.')
    h('Reported quantities and numerical criteria',2)
    p('The beam diameter is the full 1/e² intensity width along x or y, using the declared neighboring central-row convention. Useful efficiency is a Poynting-power-normalized modal quantity. To avoid selecting the more favorable normalization, the reported transmission factor is the smaller of native net transmission and reconstructed forward transmission. The frozen Gaussian E/H overlap then gives')
    eq('η = min(T_native, T_forward) × O_G.                                                            (5)')
    p('A phase-current regression of E_x within r < 0.6D_t estimates inverse wavefront radius. Its normalized curvature is C = z_R/R with z_R = πn_air(D_t/2)²/λ₀. This definition is distinct from simply plotting wrapped phase or subtracting a fitted defocus term. The angular diagnostic Q is the retained-forward-spectrum mean of k_x² + k_y² divided by the corresponding frozen Gaussian-target mean. It is an uncentred moment ratio, not its square root and not ISO M². Gaussian-core half-angles are fitted separately from the weak-angle halo. A complete beam-propagation-ratio claim would require the relevant beam-characterization methods and uncertainties [8].')
    p('Nominal criteria require transmission and useful efficiency at least 0.90, overlap at least 0.98, output diameters within 10% of the frozen vector-target crossings, sidelobe fraction no larger than 0.02, |C| ≤ 0.10, Q ≤ 1.10 and valid processing-plane crossings. The final same-settings mesh pair must change all processing diameters by less than 2% and useful efficiency by less than 5%. Each fitted core half-angle change must be no larger than max(2% of the prior angle, 0.001 rad). The curvature difference must be ≤0.005 and smaller than the remaining |C| acceptance margin. Q changes are retained as diagnostics; they were not added retrospectively as a mesh-pair convergence threshold.')
    h('3 Results')
    h('Learning behavior and the contribution of the neural correction',2)
    p('Figure 2 separates the raw sampled-height training objective from quantized validation and from final proxy comparisons. Successive raw training points generally correspond to different sampled H values, so their changes cannot be interpreted as a monotonic per-height learning curve. The fixed validation checkpoints provide the consistent selection comparison. The observed aggregate validation improvement establishes successful optimization and checkpoint selection under the stated objective; it does not imply uniform improvement at every height or superiority to the non-neural method.')
    p('The useful-efficiency changes relative to the quantized physics seed are small. At H40/H75/H100, the selected NN changes proxy useful efficiency by +0.02597, +0.00312 and −0.01475 percentage points, respectively. The selected direct-table comparator changes it by +0.02769, +0.00682 and +0.00740 percentage points. The objective balances several observables, so a lower objective need not coincide with higher useful efficiency. In particular, H100 useful efficiency is below the field-seed value, and final quantized loss at H75/H100 does not uniformly improve the seed. A matched Maxwell comparison of seed, NN and direct-table devices has not been completed for these three heights. The direct Meep curves below refer to forward validation of the NN pattern, not to the non-neural optimizer.')
    fig('fig02_learning_and_proxy_contribution','Figure 2. Learning and available evidence for the design correction. Raw training histories retain their sampled-height variation; quantized validation checkpoints evaluate a fixed height set. Seed/checkpoint outcomes and selected-height proxy comparisons are shown without replacing unfavorable outcomes. Physics seed, NN and direct-table optimizer use the same fixed optical problem and final 50 nm quantization. All performance comparisons in this figure use the differentiable proxy. H75 participated in validation. Modest gains, the H100 efficiency decrease and seed-to-seed differences do not establish a full-wave NN advantage. Raw data and checkpoint-selection records accompany the figure.')
    h('Relief profiles and output wavefronts',2)
    p('The selected H40/H75/H100 patterns occupy nonzero relief diameters of 42, 37 and 37 µm, respectively, within their common 50 µm cylinder footprint. Each reaches 3 µm maximum relief, so total assembly heights are 43, 78 and 103 µm. The 50 nm steps describe the vertical height levels; the radial annular width is 500 nm. The selected profiles are centrally connected and nonincreasing with radius, although the learning representation itself supplies bounded independent residuals rather than a universal monotonicity theorem.')
    p('Figure 3 displays the frozen profiles and fields reconstructed from the selected archived Maxwell records. Intensity is proportional to total electric-field magnitude squared after matched incident-power normalization and is shown on one common logarithmic scale. The phase panels display arg(E_x) after removing a single global piston. No curvature or defocus is subtracted. Restricting the display to the stated bright-field mask avoids assigning physical meaning to phase in regions of very low electric amplitude. The residual phase structure shows that near-flat wavefronts can coexist with quantization rings and weak field structure; the panels are not experimental micrographs.')
    fig('Figure3_profiles_and_fields_50nm',(HERE/'figures/Figure3_caption.txt').read_text(encoding='utf-8').strip(),width=6.35)
    h('Collimation at the three representative heights',2)
    p('Figure 4 compares the design proxy with the independently reconstructed Maxwell fields of the same frozen pattern at each height. At H40, H75 and H100, useful modal efficiencies are 95.28765%, 95.49539% and 95.53601%. The output diameter pairs are 11.73994 × 11.55089 µm, 14.41311 × 14.32069 µm and 16.68355 × 16.67200 µm. Their fitted Gaussian-core half-angle pairs are 4.83631°/4.84074°, 3.98196°/3.98458° and 3.44323°/3.44528°. The larger natural footprint at longer H is associated with a smaller core divergence, but the output beam is not approximately 30 µm in diameter.')
    p('At downstream distances 0, 25, 50, 75 and 100 µm, the largest proxy–Maxwell width differences, taking both axes and using Maxwell as the denominator, are 1.80%, 0.81% and 1.22% for the three heights. These values measure agreement of the two forward models for the selected geometry. They do not measure NN-versus-optimizer improvement. The curves retain the x/y width span rather than treating that span as a confidence interval. The Gaussian reference is predeclared from the natural footprint and retains its own diffractive expansion.')
    p('The Maxwell curvature values are −0.031134, −0.036628 and +0.026593 for H40/H75/H100; the corresponding Q values are 1.026930, 1.031168 and 1.027571. The proxy reproduces the beam-width evolution closely while retaining visible curvature and angular-moment bias. Across the complete eight-height assessment, useful efficiency spans 95.287654–95.558691%, |C| does not exceed 0.0366281 and Q ranges from 0.998208 to 1.049763. These observables support near-declared-Gaussian collimation under the stated definitions. A Q close to or below unity does not demonstrate sub-diffraction behavior or a global aperture limit.')
    fig('figure_4_collimation_performance','Figure 4. Forward-model comparison for the original 50 nm NN patterns. Panels a–c show x/y-mean 1/e² intensity diameters versus distance after the output monitor at H + 4 µm; shaded spans are the x/y width difference, not uncertainty bands. Blue dashed curves are proxy predictions and orange curves are independent Meep results for the same height-specific geometry. Dotted curves are the frozen scalar Gaussian references. Panels d–f show useful efficiency, normalized curvature C and retained-band angular ratio Q. The dotted Q = 1 line is the target-moment ratio, not an M² or global diffraction bound. H40/H100 are training heights; H75 is validation-observed. Meep meshes are 32, 32 and 40 cells/µm.')
    table('Table 2  Final Maxwell observables of the original 50 nm NN family',
      ['H\nµm','Resolution\ncells/µm','Dₓ\nµm','Dᵧ\nµm','η\n%','C','Q'],
      [[f"{r['H_um']:g}",f"{r['resolution_cells_per_um']:g}",f"{r['output_Dx_um']:.4f}",f"{r['output_Dy_um']:.4f}",f"{r['eta_percent']:.4f}",f"{r['C']:.6f}",f"{r['Q']:.6f}"] for r in PERF],
      [.08,.14,.15,.15,.15,.17,.16])
    p('All eight entries in Table 2 satisfy the declared native-field, nominal optical and final mesh-pair criteria. Diameters are measured at each height’s output monitor, not at the fiber facet. Efficiencies use Eq. (5). The different final meshes reflect the recorded refinement sequence and should not be concealed by describing the three cases as a single matched-grid comparison.')
    h('Numerical verification and remaining sensitivity',2)
    p('Figure 5 retains earlier nominal mesh results and failed comparisons. The final H40 and H75 pairs are 24→32 cells/µm, while H100 uses 32→40 cells/µm. Their maximum processing-width changes are 0.5291%, 0.4108% and 0.5900%, and the corresponding absolute curvature changes are 0.0001074, 0.0042893 and 0.0023773. All meet the declared last-pair criteria. Across the family, H60 and H70 also require 32→40, while H50, H80 and H90 conclude at 24→32. These are measured pairwise stability results, not an extrapolation to zero grid spacing.')
    p('The H100 retained-band Q decreases from 1.103365 at 24 cells/µm to 1.043625 at 32 and 1.027571 at 40. The nominal 1.10 criterion is therefore evaluated on the finer retained result rather than repaired by threshold relaxation or halo removal. Its 32→40 change is 0.0160535; that value remains an explicit angular diagnostic, because the frozen mesh protocol does not impose a Q-change limit. Fine-mesh acceptance of widths, efficiency and phase curvature does not establish full all-angle second-moment convergence.')
    p('Four original controls were completed at 16 cells/µm: PML changes at H40 and H100, radial-domain/padding change at H70 and duration change at H100. They pass their declared sampled comparisons. The H70 domain change also increases the native monitor radius from 35 to 40 µm, so it tests a combined domain/aperture perturbation. Increasing the H100 post-source duration from 335 to 502.5 changes useful efficiency by −0.22045 percentage points and Q by +0.00352218. This coarse result cannot be used as a correction to the fine-mesh values. Fine-grid temporal and boundary sensitivity remains a limitation when stronger convergence or complete beam-quality claims are sought.')
    fig('fig05_mesh_and_numerical_verification','Figure 5. Numerical verification from saved nominal Maxwell analyses, retaining the earlier meshes and the selected final fields. Useful efficiency, C and Q are shown across the recorded resolution sequence. The remaining panel distinguishes successive processing-width changes from native device power closure and labels their separate criteria. A nominal Q threshold is different from a convergence threshold on its mesh-to-mesh change. The final width/efficiency/core-fit/curvature pairs pass the frozen checks. Original boundary and duration controls were sampled at 16 cells/µm and are summarized separately in the archived control table; they do not establish finest-grid or all-angle convergence.')
    h('4 Interpretation and limitations')
    p('The principal demonstrated result is that a physics-seeded neural geometry family can be carried through independent Maxwell assessment and traceable geometry export while retaining explicit metric definitions. The selected surfaces compensate the phase accumulated through a finite polymer cylinder and produce near-flat output wavefronts with useful efficiency near 95%. The growing downstream widths remain consistent with finite-beam diffraction. The definition of collimation here is consequently tied to wavefront curvature, a declared Gaussian reference and fitted core divergence, rather than to an indefinitely invariant beam radius.')
    p('The current results do not establish that the neural representation is the best optimizer for this problem. The field seed already supplies most of the collimating structure, the direct-table proxy comparator is competitive, and useful-efficiency gains are small. A future algorithmic advantage claim would need matched independent Maxwell comparisons of seed, NN and non-neural designs, together with complete training/seed-construction/inference costs. The present proxy comparison is retained as an ablation of the correction, including unfavorable cases, rather than presented as evidence of a universal NN improvement.')
    p('The source and material assumptions also bound interpretation. A real fiber eigenmode, measured polymer absorption and post-processing index, wavelength dependence, axis offsets, angular misalignment, environmental variation and fabricated-surface roughness have not been validated here. Centered axisymmetry enables the m = ±1 treatment; asymmetric fabrication and offset illumination require a suitably general vector model. Initially withheld proxy heights do not substitute for independent Maxwell testing of a new height family, and H-only conditioning does not establish robustness to changed MFD or NA.')
    p('Numerical stability and geometric correctness address different error sources. The last-pair tests limit the observed changes in specified observables, but the retained spectral band and finite native aperture restrict halo-sensitive conclusions. The 128-segment circular tessellation used for SLC introduces a separate boundary approximation. Printer import, laser hatching, exposure, resin shrinkage, adhesion and optical measurements are absent. These limits should remain visible if the files are used to plan an experimental study.')
    h('5 From 50 nm relief quantization to 500 nm SLC layers')
    h('Three different lengths',2)
    p('The word spacing can refer to three independent quantities in this project. First, the radial annular width is 0.5 µm, or 500 nm, for both design versions. Second, the original surface height is selected on a 0.05 µm, or 50 nm, vertical grid. Third, the SLC layer thickness is the vertical distance between consecutive cross-section records. The original SLC uses 0.05 µm layers, whereas the later fabrication export uses 0.5 µm layers. Changing the third quantity does not turn an arbitrary 50 nm staircase into an unchanged surface with fewer labels.')
    p('In plain language, the original lens can finish a ring at many closely spaced height levels. A printer job restricted to 500 nm layers can finish that ring only at every tenth original level. The export therefore moves each ring to the nearest allowed coarse level and then slices that new solid. The cylinder height, cylinder diameter, optical-axis position and unit scale are preserved; the fine top surface is changed.')
    h('Exact height conversion and tie handling',2)
    p('The exporter first recovers the nominal integer index of the original 50 nm level. This avoids interpreting a float32 representation such as 2.250000… µm as a different side of a coarse rounding tie. It then rounds to the nearest 500 nm level with exact nominal ties rounded upward. For nonnegative annular heights, the implemented rule is')
    eq('n_j = rint[h₅₀,j/(0.05 µm)],      m_j = floor[(n_j + 5)/10],      h₅₀₀,j = 0.5 µm × m_j.     (6)')
    p('Equation (6) is the actual coarse-export rule, not a direct floating-point round(h/0.5). Its tie convention is also distinct from the nearest-even convention used in the original training quantizer. For example, 2.20 µm becomes 2.00 µm, 2.25 µm becomes 2.50 µm, 2.30 µm becomes 2.50 µm and 0.10 µm becomes zero relief. The theoretical nearest-level deviation is at most 0.25 µm, and that maximum is reached by the exported profiles. The fine profile and its original SLC remain preserved as separate records.')
    p('The changes affect 35 of 50 annuli at H40 and 32 annuli at each of H75/H100. Ring-wise RMS height errors, counting all 50 equal-width annuli, are 0.12490, 0.11747 and 0.11683 µm. The nonzero relief footprints shrink from 42/37/37 µm to 41/36/36 µm, while the full cylinder footprint remains 50 µm. These changes are shown in Supplementary Figure S1. A zero-relief outer ring still contains the intended cylinder below H; it is not deleted from the full assembly.')
    h('Layer construction coordinates and file units',2)
    p('The fabrication solid is rebuilt as H + h₅₀₀(r), with the cylinder and relief joined into one material body. For a slab [z_i, z_(i+1)), an interior sample supplies its XY contour and z_i is stored as the record’s minimum Z. In nominal micrometre coordinates, z_i = iΔz and the interior sample is approximately z_i + Δz/2. Float32 serialization introduces small numerical offsets, so measured layer gaps are reported separately from the nominal requested spacing. Storing the sample midpoint as the record boundary would shift the reconstructed solid and is not the convention used.')
    eq('Δz₅₀ = 0.05 µm = 0.00005 mm,      Δz₅₀₀ = 0.5 µm = 0.0005 mm.                          (7)')
    p('Both SLC versions serialize physical millimetres. Mesh coordinates are converted from micrometres to millimetres exactly once, with XY and Z scale overrides of 0.001 mm per source micrometre. The cylinder therefore remains 0.050 mm in diameter. The factor-of-ten change in layer count is a physical coarsening in Z, not a factor-of-ten scale change of the lens. The current user-supplied builder already writes minimum-Z records; its native output is retained without applying the historical midpoint relabeling a second time.')
    p('All selected cylinders have heights divisible by 0.5 µm and retain a maximum relief of 3 µm. Thus total heights stay at 43, 78 and 103 µm. The fine exports have 860, 1560 and 2060 slabs, while the coarse exports have 86, 156 and 206. Within those counts, the full cylinders occupy 800/1500/2000 fine slabs or 80/150/200 coarse slabs, and the top relief occupies 60 fine slabs or six coarse slabs. The terminal top record is not an additional material layer.')
    table('Table 3  Original and coarse fabrication geometry',
      ['H\nµm','Total top\nµm','Fine slabs\n50 nm','Coarse slabs\n500 nm','Relief diameter\nfine → coarse µm','Changed\nannuli','RMS height\nerror µm'],
      [[str(int(r['H_um'])),f"{r['original_50nm']['assembly_total_height_um']:g}",str(r['original_50nm']['layer_count']),str(r['derived_500nm']['layer_count']),f"{r['original_50nm']['nonzero_footprint_diameter_um']:g} → {r['derived_500nm']['nonzero_footprint_diameter_um']:g}",str(r['change']['changed_annuli']),f"{r['change']['unweighted_ring_height_RMS_error_um']:.5f}"] for r in FAB],
      [.07,.11,.12,.13,.25,.12,.20])
    h('What the SLC audit establishes',2)
    p('Independent parsing checks the header and MM units, finite closed contour vertices, winding, absence of self-intersections, base and top coordinates, layer count and spacing, complete termination and source-to-solid correspondence. The connected exterior mesh is checked for manifoldness, orientation and one-body connectivity. Full forward-held solids are compared by symmetric-difference volume, so added and missing material cannot cancel as they could in a signed-volume-only test. Circular tessellation error is reported separately: the 128-gon boundary has a maximum sagitta of 7.5295 nm at the 25 µm outer radius and a 0.0401547% volume deficit relative to ideal circular sections.')
    p('The coarse files pass geometry checks against their declared coarse fabrication sources. They correctly fail strict unchanged-source checks against the fine relief. Whole-assembly circular symmetric-difference fractions between coarse and fine versions are 0.21716%, 0.09264% and 0.06837%. These percentages use the volume of the entire cylinder plus relief as denominator; the large unchanged cylinder makes them small. They do not imply that a 250 nm top-surface error is optically negligible.')
    p('No electromagnetic calculation in this manuscript revalidates the 500 nm profiles. Therefore the efficiencies, beam widths, C values and Q values in Figures 3–5 and Table 2 belong only to the original 50 nm geometries. The coarse SLC demonstrates a specified, traceable fabrication representation. Its optical performance must be calculated or measured independently. Likewise, commanded 50 nm or 500 nm layer spacing is not a measurement of a two-photon printer’s voxel size or achievable resolution, and an SLC contour file is not a calibrated laser-power, scan-speed or hatch-path recipe.')
    h('6 Conclusions')
    p('We have assessed a height-conditioned, physics-seeded neural inverse-design workflow for fiber-mounted polymer collimators at 1550 nm. The original 50 nm-quantized NN family passes its declared independent native-field, nominal and last-mesh-pair tests at eight assessed heights. The representative 40, 75 and 100 µm cylinders provide useful modal efficiencies near 95.3–95.5%, low normalized phase curvature and propagation close to their predeclared natural-footprint Gaussian references. Independent Maxwell checking confirms the reported behavior of the selected NN profiles while exposing forward-model differences in curvature and retained angular moments.')
    p('The evidence does not establish a general NN advantage, arbitrary-input generalization, experimental performance or a global diffraction limit. A separate, audited 500 nm SLC export reduces layer counts by ten through explicit nearest-level requantization of the relief. That process preserves cylinder dimensions and coordinates while changing the optical surface by up to 250 nm. Maintaining separate fine optical and coarse fabrication records is essential to interpreting both the numerical results and any later printing experiment.')
    h('Data and reproducibility')
    p('The study retains the configurations, selected checkpoint lineage, original profiles, complex fields, matched references, numerical analyses, plotting arrays, figure CSVs and SLC manifests in the project evidence archive. The companion manuscript_facts.json and figure provenance files identify the numeric sources and SHA256 hashes. Appendix A gives the principal relative locations. No public repository accession is asserted in this document. Experimental images or measurements are not part of the dataset.')
    h('References')
    refs=[
      '[1] S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vučković and A. W. Rodriguez, Inverse design in nanophotonics, Nature Photonics 12, 659–670 (2018). https://doi.org/10.1038/s41566-018-0246-9',
      '[2] J. Jiang and J. A. Fan, Global optimization of dielectric metasurfaces using a physics-driven neural network, Nano Letters 19, 5366–5372 (2019). https://doi.org/10.1021/acs.nanolett.9b01857',
      '[3] T. Gissibl, S. Thiele, A. Herkommer and H. Giessen, Sub-micrometre accurate free-form optics by three-dimensional printing on single-mode fibres, Nature Communications 7, 11763 (2016). https://doi.org/10.1038/ncomms11763',
      '[4] K. Vanmol, A. A. Abdul Nazar, H. Thienpont, F. Ferranti and J. Van Erps, Fabrication of multilevel metalenses using multiphoton lithography from design to evaluation, Optics Express 32, 10190–10203 (2024). https://doi.org/10.1364/OE.514237',
      '[5] Corning Incorporated, Corning SMF-28 Ultra Optical Fiber Product Information, PI-1424-AEN, July 2025. https://www.corning.com/media/worldwide/coc/documents/Fiber/product-information-sheets/PI-1424-AEN.pdf',
      '[6] D. P. Kingma and J. Ba, Adam A method for stochastic optimization, International Conference on Learning Representations (2015). https://arxiv.org/abs/1412.6980',
      '[7] A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos and S. G. Johnson, Meep A flexible free-software package for electromagnetic simulations by the FDTD method, Computer Physics Communications 181, 687–702 (2010). https://doi.org/10.1016/j.cpc.2009.11.008; solver documentation https://meep.readthedocs.io/',
      '[8] International Organization for Standardization, ISO 11146-1:2021 Lasers and laser-related equipment Test methods for laser beam widths divergence angles and beam propagation ratios Part 1 Stigmatic and simple astigmatic beams (2021).'
    ]
    for s in refs:blocks.append(dict(kind='ref',text=s))
    h('Appendix A Evidence index')
    p('Paths below are relative to the fiber_metalens_nn project root. The latest scientific assessment uses additive recovery of the final H100 saved fields; the original attempt-83 controller exit/completion fields remain unrecorded and the original workflow clock is not promoted to a pass. Scientific saved-field verification and execution bookkeeping are retained separately. Earlier failed meshes, historical exports and raw files remain preserved.')
    for name,path in [
      ('Physical inputs','artifacts/1550nm_height_family_n15263_v2/config.json'),
      ('Learning audit','artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v2/learning_evidence_v1/'),
      ('Final optical assessment','artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v3/assessment/recovery_final_verification_v2/'),
      ('Final three-height comparison','artifacts/1550nm_height_family_n15263_v2/maxwell_validation_v3/forward_comparison/recovery_final_v1/'),
      ('Original 50 nm SLC audit','exports/height_family_cylinder_attached_slc_v1/VERIFICATION.txt'),
      ('Coarse 500 nm SLC audit','exports/height_family_cylinder_attached_slc_500nm_v5/VERIFICATION.txt'),
      ('Manuscript figures and numeric arrays','reports/manuscript_50nm_v1/figures/'),
      ('Authoring facts and source hashes','reports/manuscript_50nm_v1/evidence/manuscript_facts.json')]:
        blocks.append(dict(kind='path',text=name+'\n'+path))
    h('Supplementary profile comparison')
    fig('figure_S1_quantization_conversion','Supplementary Figure S1. Actual relief profiles before and after the separate 500 nm export. Blue curves retain the original 50 nm-quantized optical profiles; orange curves are the nearest-500 nm fabrication profiles with nominal ties rounded upward. Radial annular width remains 0.5 µm in both versions. Cylinder height and diameter remain unchanged, while maximum height error reaches 0.25 µm and the outer nonzero relief diameter decreases. The orange curves have geometry validation only; optical metrics from the original profiles are not assigned to them.')
    return blocks

def set_cell_shading(cell,color):
    pr=cell._tc.get_or_add_tcPr();el=OxmlElement('w:shd');el.set(qn('w:fill'),color);pr.append(el)
def font(run,size=11,bold=False):
    run.font.name='Times New Roman';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor(0,0,0)
    run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'),'Times New Roman');run._element.rPr.rFonts.set(qn('w:hAnsi'),'Times New Roman')
def render():
    doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
    sec.top_margin=sec.bottom_margin=Inches(.73);sec.left_margin=sec.right_margin=Inches(.80)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
        st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
        st.element.get_or_add_rPr().rFonts.set(qn('w:ascii'),'Times New Roman');st.element.rPr.rFonts.set(qn('w:hAnsi'),'Times New Roman')
    normal=doc.styles['Normal'];normal.font.size=Pt(11);normal.paragraph_format.line_spacing=1.10;normal.paragraph_format.space_after=Pt(6)
    for name,size in [('Heading 1',13),('Heading 2',11.5)]:
        st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True;st.paragraph_format.space_before=Pt(11);st.paragraph_format.space_after=Pt(5)
    doc.styles['Title'].font.size=Pt(21);doc.styles['Title'].paragraph_format.space_after=Pt(7)
    doc.styles['Subtitle'].font.size=Pt(12);doc.styles['Subtitle'].paragraph_format.space_after=Pt(14)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
    for block in blocks:
        kind=block['kind']
        if kind in ('title','subtitle'):
            doc.add_paragraph(block['text'],style='Title' if kind=='title' else 'Subtitle')
        elif kind=='h':doc.add_heading(block['text'],level=block['level'])
        elif kind in ('p','ref','path','eq'):
            para=doc.add_paragraph()
            if kind=='eq':para._p.append(native_math.equation(block['text']))
            elif kind=='p':native_math.add_inline(para,block['text'],lambda value:font(para.add_run(value),11))
            else:font(para.add_run(block['text']),10)
            if kind=='ref':para.paragraph_format.space_after=Pt(7);para.paragraph_format.line_spacing=1.05
            if kind=='path':para.paragraph_format.space_after=Pt(8);para.paragraph_format.line_spacing=1.05
            if kind=='eq':
                para.alignment=WD_ALIGN_PARAGRAPH.CENTER;para.paragraph_format.space_before=Pt(4);para.paragraph_format.space_after=Pt(8)
        elif kind=='figure':
            path=HERE/'figures'/(block['stem']+'.png')
            if not path.exists():raise FileNotFoundError(path)
            para=doc.add_paragraph();para.alignment=WD_ALIGN_PARAGRAPH.CENTER;para.paragraph_format.keep_with_next=True;para.paragraph_format.space_before=Pt(8)
            para.add_run().add_picture(str(path),width=Inches(block['width']))
            cap=doc.add_paragraph();cap.paragraph_format.line_spacing=1.05;cap.paragraph_format.space_after=Pt(10)
            cap.paragraph_format.keep_together=True
            text=block['caption'];split=text.find('. ')
            if split>=0:font(cap.add_run(text[:split+1]+' '),9.5,True);font(cap.add_run(text[split+2:]),9.5)
            else:font(cap.add_run(text),9.5)
        elif kind=='table':
            cap=doc.add_paragraph(block['caption']);cap.paragraph_format.keep_with_next=True
            if block['caption'].startswith('Table 2'):cap.paragraph_format.page_break_before=True
            for r in cap.runs:font(r,10,True)
            tb=doc.add_table(rows=1,cols=len(block['headers']));tb.alignment=WD_TABLE_ALIGNMENT.CENTER;tb.autofit=False
            widths=block['widths'] or [1/len(block['headers'])]*len(block['headers']);norm=sum(widths)
            for i,w in enumerate(widths):tb.columns[i].width=Inches(6.60*w/norm)
            for i,value in enumerate(block['headers']):tb.rows[0].cells[i].text=value
            repeat=OxmlElement('w:tblHeader');tb.rows[0]._tr.get_or_add_trPr().append(repeat)
            for row in block['rows']:
                cells=tb.add_row().cells
                for i,value in enumerate(row):cells[i].text=str(value)
            pr=tb._tbl.tblPr;borders=OxmlElement('w:tblBorders')
            for name in ['top','left','bottom','right','insideH','insideV']:
                edge=OxmlElement('w:'+name);edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'D9D9D9');borders.append(edge)
            pr.append(borders)
            for j,row in enumerate(tb.rows):
                cant=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(cant)
                for i,cell in enumerate(row.cells):
                    cell.width=Inches(6.60*widths[i]/norm);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    set_cell_shading(cell,'DFEAF2' if j==0 else ('FFFFFF' if j%2 else 'F4F6F8'))
                    marg=OxmlElement('w:tcMar')
                    for side in ['top','left','bottom','right']:
                        el=OxmlElement('w:'+side);el.set(qn('w:w'),'90');el.set(qn('w:type'),'dxa');marg.append(el)
                    cell._tc.get_or_add_tcPr().append(marg)
                    for para in cell.paragraphs:
                        para.paragraph_format.space_after=Pt(0);para.paragraph_format.space_before=Pt(0);para.paragraph_format.line_spacing=1.05
                        para.alignment=WD_ALIGN_PARAGRAPH.LEFT if len(row.cells)==2 else WD_ALIGN_PARAGRAPH.CENTER
                        for run in para.runs:font(run,9.3,j==0)
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
    # Remove inherited paragraph rules, including the default blue Title rule.
    for parent in (doc.element,doc.styles.element):
        for border in list(parent.xpath('.//w:pBdr')):border.getparent().remove(border)
    doc.core_properties.title='Physics seeded neural inverse design of fiber mounted polymer collimators'
    doc.core_properties.subject='Original 50 nm computational design and separate 500 nm SLC geometry conversion'
    doc.core_properties.author='';doc.core_properties.keywords='fiber collimation neural inverse design 50 nm 500 nm SLC'
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():raise FileExistsError('Preserve existing manuscript; choose a fresh file before rebuilding')
    doc.save(OUT)
    (HERE/'docx_build_manifest.json').write_text(json.dumps({'output':str(OUT),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'blocks':len(blocks),'embedded_figures':[b['stem'] for b in blocks if b['kind']=='figure'],'facts_sha256':hashlib.sha256((HERE/'evidence/manuscript_facts.json').read_bytes()).hexdigest(),'new_optical_solves':0},indent=2)+'\n')
    print(OUT)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--text-only',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.output:OUT=args.output.resolve()
    content()
    corrections={'160, 260 and 335':'160, 259 and 335','[2⟨r²⟩_I]¹ᐟ²':'sqrt(2⟨r²⟩_I)','w_t(z)[ln(100)/2]¹ᐟ²':'w_t(z) sqrt[ln(100)/2]',
      'lithography from design to evaluation, Optics':'lithography: from design to evaluation, Optics',
      'Adam A method for stochastic optimization':'Adam: A method for stochastic optimization',
      'Meep A flexible free-software':'Meep: A flexible free-software',
      'neighboring-height variation':'neighboring annular-height variation',
      'Its envelope and exaggerated relief are schematic, not simulated intensity data.':'Its envelope and exaggerated relief are schematic and not to scale, not simulated intensity data.',
      'from native device power closure':'from the larger of native device and reference power-closure errors'}
    for block in blocks:
        for key in ('text','caption'):
            if key in block:
                for old,new in corrections.items():block[key]=block[key].replace(old,new)
    for block in blocks:
        if block['kind']=='p':
            block['text']=block['text'].replace('D_t = 2sqrt(2⟨r²⟩_I)','⟦TARGET_DIAMETER⟧').replace('w_t(z) sqrt[ln(100)/2]','⟦SIDE_RADIUS⟧').replace('z_R = πn_air(D_t/2)²/λ₀','⟦RAYLEIGH⟧').replace('z_R/R','⟦CURVATURE⟧')
        if block['kind']=='figure' and block['stem']=='Figure3_profiles_and_fields_50nm':
            block['caption']='Figure 3. Frozen 50 nm quantized profiles and output fields from archived Maxwell records 43, 69 and 83, corresponding to H = 40, 75 and 100 µm. Panels a–c show the radial relief. Panels d–f show total electric intensity after matched incident-power normalization; one peak across the three panels defines the common logarithmic scale. Panels g–i display arg(Eₓ) after removing one intensity-weighted global piston, without curvature or defocus subtraction. Phase is displayed only where total electric intensity and Eₓ intensity each exceed 1% of their own lens-specific peaks; gray pixels are masked. All maps are at z = H + 4 µm in air. The saved reconstruction uses a 512 × 512 grid over 128 µm, centered linear polarization, the paraxial Yee power-equivalent correction and the retained forward spectral band. The panels crop x and y to ±25 µm. These are saved-data visualizations of the original 50 nm profiles, not experimental measurements or optical validation of the 500 nm export.'
    # Explicit first-use callouts complement the detailed captions.
    for block in blocks:
        if block['kind']=='p' and block['text'].startswith('The facet is z = 0.'):
            block['text']+=' Figure 1 defines these coordinates, and Table 1 collects the fixed inputs.'
        if block['kind']=='p' and block['text'].startswith('All selected cylinders have heights divisible'):
            block['text']+=' Table 3 summarizes the dimensions and layer counts.'
        if block['kind']=='figure':
            block['caption']=block['caption'].replace('at z = H + 4 µm in air','at z\u00a0=\u00a0H\u00a0+\u00a04\u00a0µm in air')
    (HERE/'manuscript_content.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Words',sum(len((b.get('text','')+' '+b.get('caption','')).split()) for b in blocks))
    if not args.text_only:render()
