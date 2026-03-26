# td_calc.py  —— One kick for ONE system, records dm/mm
#gpaw -P 32 python td_calc.py -- --gpw ./work_Ag4/Ag4.gpw --mode lcao --gauge length --kick x --outdir ./work_Ag4
#gpaw -P 64 python td_calc.py -- --gpw ./work_Ag4/Ag4.gpw --mode lcao --gauge length --kick y --outdir ./work_Ag4
#gpaw -P 64 python td_calc.py -- --gpw ./work_Ag4/Ag4.gpw --mode lcao --gauge length --kick z --outdir ./work_Ag4
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter
from gpaw.tddft import TDDFT as FD_TDDFT, DipoleMomentWriter as FD_DM, MagneticMomentWriter as FD_MM
import argparse, os, inspect


def run_lcao(gpw, outdir, kick, gauge, dt_as, nsteps, origin, origin_shift):
    os.makedirs(outdir, exist_ok=True)
    td = LCAOTDDFT(gpw, txt=os.path.join(outdir, f'td-{kick}.out'))

    # 兼容旧 API：检查 absorption_kick 是否支持 gauge 形参
    sig = inspect.signature(td.absorption_kick)
    supports_gauge = ('gauge' in sig.parameters)
    real_gauge = gauge if supports_gauge else 'length'  # 旧版只能用 length

    # 写偶极与磁矩（文件名带上“实际使用”的规）
    DipoleMomentWriter(td, os.path.join(outdir, f'dm-{real_gauge}_{kick}.dat'))
    if origin == 'COM':
        MagneticMomentWriter(td, os.path.join(outdir, f'mm-COM-{real_gauge}_{kick}.dat'),
                             origin='COM')
    elif origin == 'zero':
        MagneticMomentWriter(td, os.path.join(outdir, f'mm-ZERO-{real_gauge}_{kick}.dat'),
                             origin='zero', origin_shift=origin_shift)
    else:
        MagneticMomentWriter(td, os.path.join(outdir, f'mm-{origin}-{real_gauge}_{kick}.dat'),
                             origin='COM', origin_shift=origin_shift)

    # δ-kick
    k = [0.0, 0.0, 0.0]
    k['xyz'.index(kick)] = 1e-5
    if supports_gauge:
        td.absorption_kick(k, gauge=gauge)
    else:
        print(f"[INFO] This GPAW LCAO TDDFT does not accept 'gauge='. "
              f"Falling back to length gauge. (requested='{gauge}')")
        td.absorption_kick(k)

    td.propagate(dt_as, nsteps)
    td.write(os.path.join(outdir, f'td-{kick}.gpw'), mode='all')

def run_fd(gpw, outdir, kick, dt_as, nsteps):
    os.makedirs(outdir, exist_ok=True)
    td = FD_TDDFT(gpw, solver=dict(name='CSCG', tolerance=1e-8),
                  txt=os.path.join(outdir, f'td-{kick}.out'))
    FD_DM(td, os.path.join(outdir, f'dm-{kick}.dat'))
    FD_MM(td, os.path.join(outdir, f'mm-{kick}.dat'))
    kick_strength = [0.0, 0.0, 0.0]
    kick_strength['xyz'.index(kick)] = 1e-5
    td.absorption_kick(kick_strength)
    td.propagate(dt_as, nsteps)
    td.write(os.path.join(outdir, f'td-{kick}.gpw'), mode='all')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpw", required=True, help="上一阶段产生的 .gpw 文件 (ground-state file)")
    ap.add_argument("--mode", default="lcao", choices=["lcao","fd"])
    ap.add_argument("--kick", default="x", choices=["x","y","z"])
    ap.add_argument("--gauge", default="length", choices=["length","velocity"], help="仅 LCAO 有效 (gauge)")
    ap.add_argument("--dt_as", type=float, default=10.0, help="步长 attoseconds")
    ap.add_argument("--nsteps", type=int, default=1600, help="步数 steps")
    ap.add_argument("--origin", default="COM", help="磁矩原点 (origin): COM/zero/自定义名")
    ap.add_argument("--origin_shift", type=float, nargs=3, default=[0.0,0.0,0.0])
    ap.add_argument("--outdir", default="rt_out")
    args = ap.parse_args()

    if args.mode == 'lcao':
        run_lcao(args.gpw, args.outdir, args.kick, args.gauge, args.dt_as, args.nsteps,
                 args.origin, args.origin_shift)
    else:
        run_fd(args.gpw, args.outdir, args.kick, args.dt_as, args.nsteps)

