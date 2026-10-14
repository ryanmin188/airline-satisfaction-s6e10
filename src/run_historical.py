"""Run preserved historical source with only absolute paths remapped locally."""
import runtime
from runtime import ROOT
import ast, argparse, sys
ap=argparse.ArgumentParser();ap.add_argument('script',choices=['feature_5_conditional_frequency.py','mlp_lean_1.py','tabm_m3_full_oof.py','tabm_m3_multiseed.py']);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
dest=ROOT/'retraining/source';dest.mkdir(parents=True,exist_ok=True)
class Paths(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value.startswith('D:\\kaggle\\airline'):
            suffix=node.value[len('D:\\kaggle\\airline'):].lstrip('\\').replace('\\','/')
            target=ROOT/suffix if suffix=='data' or suffix.startswith('data/') else ROOT/'retraining'/suffix
            return ast.copy_location(ast.Constant(str(target)),node)
        return node
for source in (ROOT/'src/historical').glob('*.py'):
    tree=Paths().visit(ast.parse(source.read_text(encoding='utf-8-sig')))
    if source.name=='tabm_m3_multiseed.py':
        # Its historical regex relies on the old Windows path and formatting.
        # Replace that launcher with assignment-level edits; preserve model code.
        replacement=ast.parse('''
def run_existing_m3_source(model_seed):
    import ast
    tree=ast.parse(SOURCE_SCRIPT.read_text(encoding='utf-8-sig'))
    counts={'OUTPUT_DIR':0,'fold_seed':0}
    class Configure(ast.NodeTransformer):
        def visit_Assign(self,node):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in counts:
                    counts[target.id]+=1
                    if target.id=='OUTPUT_DIR':node.value=ast.Constant(str(OUTPUT_DIR / f'seed_{model_seed}'))
                    else:node.value=ast.BinOp(left=ast.Constant(model_seed),op=ast.Add(),right=ast.Name(id='fold_num',ctx=ast.Load()))
            return self.generic_visit(node)
    tree=Configure().visit(tree)
    assert counts=={'OUTPUT_DIR':1,'fold_seed':1},counts
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE_SCRIPT),'exec'),{'__name__':'__main__','__file__':str(SOURCE_SCRIPT)})
''').body[0]
        tree.body=[replacement if isinstance(n,ast.FunctionDef) and n.name=='run_existing_m3_source' else n for n in tree.body]
    (dest/source.name).write_text(ast.unparse(ast.fix_missing_locations(tree)),encoding='utf-8')
    compile(tree,str(source),'exec')
if args.prepare_only:
    print('Prepared portable historical scripts:',dest)
    raise SystemExit(0)
sys.path.insert(0,str(dest))
import runpy
runpy.run_path(str(dest/args.script),run_name='__main__')
