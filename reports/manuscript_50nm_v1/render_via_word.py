"""Use Word PDF conversion and the unchanged packaged DOCX PNG renderer."""
from pathlib import Path
import argparse,importlib.util,sys,tempfile
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('document',type=Path);parser.add_argument('--qa',type=Path,required=True);parser.add_argument('--renderer',type=Path,required=True,help='Path to the DOCX renderer script');args=parser.parse_args()
package=args.renderer.resolve()
temp=HERE/'tmp';temp.mkdir(exist_ok=True);tempfile.tempdir=str(temp)
spec=importlib.util.spec_from_file_location('packaged_docx_renderer',package);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pdf=args.qa/(args.document.stem+'.pdf')
if not pdf.exists():raise FileNotFoundError(pdf)
def existing_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    return str(pdf.resolve()),'Microsoft Word native PDF export; bundled LibreOffice is not available on Windows.'
module.convert_to_pdf=existing_pdf
sys.argv=[str(package),str(args.document),'--output_dir',str(args.qa),'--dpi','140','--emit_pdf']
module.main()
