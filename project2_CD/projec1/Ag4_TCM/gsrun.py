# gsrun.py — version compatible across GPAW versions
#gpaw -P 4 python gsrun.py -- --xyz ./Ag4.xyz   --mode lcao --basis dzp --nbands 48 --outdir ./work_Ag4     
from ase.io import read
from gpaw import GPAW, setup_paths
from gpaw.occupations import FermiDirac
import argparse, os, ast

def main(xyz, mode, basis, nbands, h, vacuum, outdir):
    setup_paths.insert(0, '.')  # 让 GPAW 在当前目录也找基组/伪势
    name = os.path.splitext(os.path.basename(xyz))[0]
    os.makedirs(outdir, exist_ok=True)

    atoms = read(xyz)
    atoms.center(vacuum=vacuum)

    # ✅ 跨版本稳妥：用“名字字典”启用 MomentCorrection 包裹 fast 求解器
    pois = {
        'name': 'MomentCorrectionPoissonSolver',
        'poissonsolver': 'fast',
        'moment_corrections': 1 + 3 + 5
    }

    calc = GPAW(
        mode=mode,                         # 'lcao' 或 'fd'
        basis=basis if mode == 'lcao' else None,
        h=h if mode == 'fd' else None,
        xc='PBE',
        nbands=nbands,
        occupations=FermiDirac(0.0),
        poissonsolver=pois,                # 这里传 dict
        convergence={'density': 1e-12},
        txt=os.path.join(outdir, f'{name}-gs.out'),
        symmetry={'point_group': False},
        parallel=dict(sl_auto=True),
    )

    atoms.calc = calc                     # 新写法（替代 set_calculator）
    atoms.get_potential_energy()          # 触发 SCF
    gpw = os.path.join(outdir, f'{name}.gpw')
    calc.write(gpw, mode='all')
    print('GS written:', gpw)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", required=True)
    ap.add_argument("--mode", default="lcao", choices=["lcao","fd"])
    ap.add_argument("--basis", default="dzp", help="LCAO 基组（如 'dzp' 或 {'Ag':'dzp'}）")
    ap.add_argument("--nbands", type=int, default=48)
    ap.add_argument("--h", type=float, default=0.30)
    ap.add_argument("--vacuum", type=float, default=8.0)
    ap.add_argument("--outdir", default="gs_out")
    args = ap.parse_args()

    basis = args.basis
    if isinstance(basis, str) and basis.startswith("{"):
        basis = ast.literal_eval(basis)  # 安全解析简单 dict

    main(args.xyz, args.mode, basis, args.nbands, args.h, args.vacuum, args.outdir)

