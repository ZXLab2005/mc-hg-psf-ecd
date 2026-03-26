#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian-pulse RT-TDDFT for GG-Ag2:
 - drive with GaussianPulse
 - record electric dipole, magnetic dipole, and Mulliken atomic charges

Usage (example):
export OMP_NUM_THREADS=1
gpaw -P 64 python td_gaussian_GG-Ag2.py   --gpw ./GG-Ag2.gpw   --outdir ./work_GG-Ag2_gauss   --E0_va 1e-3 --omega_eV 4.37 --sigma_fs 4.0 --t0_fs 8.0   --dir x --dt_as 20 --nsteps 1600 --origin COM
"""

import os, argparse
import numpy as np
from pathlib import Path

from ase.units import Hartree, Bohr

from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.magneticmomentwriter import MagneticMomentWriter
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.laser import GaussianPulse
from gpaw.tddft.units import au_to_fs  # 1 a.u. time = 0.02418884 fs

# ---------- helpers for Mulliken charges ----------
class DensityMatrix:
    def __init__(self, paw):
        self.wfs = paw.wfs
        self.density = paw.density
        self.using_blacs = self.wfs.ksl.using_blacs
        self.tag = None

    def _calculate_density_matrix(self, wfs, kpt):
        if self.using_blacs:
            return wfs.ksl.calculate_blocked_density_matrix(kpt.f_n, kpt.C_nM)
        else:
            rho_MM = wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM)
            wfs.bd.comm.sum(rho_MM, root=0)
            return rho_MM

    def get_density_matrix(self, tag=None):
        if tag is None or tag != self.tag:
            self.rho_uMM = []
            for kpt in self.wfs.kpt_u:
                rho_MM = self._calculate_density_matrix(self.wfs, kpt)
                self.rho_uMM.append(rho_MM)
            self.tag = tag
        return self.rho_uMM

def build_bf2atom(wfs):
    bf2atom_list = []
    for a, setup in enumerate(wfs.setups):
        bf2atom_list.extend([a] * setup.nao)
    return np.array(bf2atom_list, int)

class MullikenObserver:
    """Save per-atom Mulliken charge increment dq(t) = q(t) - q_eq."""
    def __init__(self, td_calc, dmat, bf2atom, S_MM, q_eq,
                 out_prefix="mulliken"):
        self.td_calc = td_calc
        self.dmat = dmat
        self.bf2atom = bf2atom
        self.S_MM = S_MM
        self.q_eq = q_eq
        self.out_prefix = out_prefix
        self.time_list = []
        self.dq_list = []

    def __call__(self, niter=None):
        # TD time is in atomic units -> convert to fs
        t_fs = self.td_calc.time * au_to_fs
        rho_MM = self.dmat.get_density_matrix()[0]
        if rho_MM.shape[1] != self.S_MM.shape[0]:
            rho_MM = rho_MM.T
        rhoS = np.dot(rho_MM, self.S_MM)
        natoms = len(self.q_eq)
        q_t = np.zeros(natoms)
        for mu in range(rhoS.shape[0]):
            a = self.bf2atom[mu]
            q_t[a] += rhoS[mu, mu].real
        dq = q_t - self.q_eq
        self.time_list.append(t_fs)
        self.dq_list.append(dq)

    def finalize(self):
        arr_t = np.array(self.time_list)
        arr_dq = np.array(self.dq_list)
        np.savez(f"{self.out_prefix}.npz", time_fs=arr_t, dq_t=arr_dq)
        with open(f"{self.out_prefix}.txt", "w") as f:
            f.write("# time(fs) dq_0 dq_1 ...\n")
            for i, t_fs in enumerate(arr_t):
                line = [f"{t_fs:.6f}"] + [f"{x:.8e}" for x in arr_dq[i]]
                f.write("  ".join(line) + "\n")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpw", required=True, help="Ground-state .gpw for Ag4")
    ap.add_argument("--outdir", default="./rt_out_gauss")
    # Gaussian pulse params
    ap.add_argument("--E0_va", type=float, default=5e-4, help="Field amplitude in V/? (linear regime 1e-4~1e-3)")
    ap.add_argument("--omega_eV", type=float, default=4.73, help="Carrier photon energy (eV)")
    ap.add_argument("--sigma_fs", type=float, default=4.0, help="Gaussian sigma (fs), FWHM≈2.355*sigma")
    ap.add_argument("--t0_fs", type=float, default=8.0, help="Pulse center time (fs)")
    ap.add_argument("--dir", choices=list("xyz"), default="z", help="Polarization axis")
    # time propagation
    ap.add_argument("--dt_as", type=float, default=20.0, help="Time step (attoseconds)")
    ap.add_argument("--nsteps", type=int, default=1600, help="Number of time steps")
    # magnetic moment origin
    ap.add_argument("--origin", default="COM", choices=["COM","zero"], help="Magnetic moment origin")
    ap.add_argument("--origin_shift", type=float, nargs=3, default=[0.0,0.0,0.0], help="Shift if origin='zero'")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Build GaussianPulse + uniform field ----
    # Unit conversion:
    #   1 V/? ≈ 0.0194469 a.u. (electric field)
    #   time(fs) -> a.u.:  t_au = t_fs / au_to_fs
    AU_PER_VA = 1.0 / 51.422067  # ≈ 0.0194469
    E0_au = args.E0_va * AU_PER_VA
    t0_au = args.t0_fs *au_to_fs
    sigma_au = args.sigma_fs * au_to_fs

    pulse = GaussianPulse(E0_au, t0_au, args.omega_eV, sigma_au, 'sin')
    ext = ConstantElectricField(Hartree / Bohr, [1.0 if c==args.dir else 0.0 for c in 'xyz'])
    td_potential = {'laser': pulse, 'ext': ext}

    # dump pulse profile for reference
    tgrid_fs = np.linspace(0.0, max(args.t0_fs + 6*args.sigma_fs, args.dt_as*args.nsteps*1e-3), 500)
    pulse.write(str(outdir / "pulse.dat"), tgrid_fs)

    # ---- Build TD calculator (LCAO-TDDFT) ----
    parallel = {'sl_auto': True, 'band': 2, 'domain': 32, 'augment_grids': True}
    td = LCAOTDDFT(args.gpw, td_potential=td_potential,
                   parallel=parallel, txt=str(outdir / f'td-{args.dir}.out'))

    # ---- Attach dipole & magnetic moment writers ----
    DipoleMomentWriter(td, str(outdir / f'dm-gauss_{args.dir}.dat'))
    if args.origin == 'COM':
        MagneticMomentWriter(td, str(outdir / f'mm-COM-gauss_{args.dir}.dat'), origin='COM')
    else:
        MagneticMomentWriter(td, str(outdir / f'mm-ZERO-gauss_{args.dir}.dat'),
                             origin='zero', origin_shift=args.origin_shift)

    # ---- Prepare Mulliken charge observer ----
    dmat = DensityMatrix(td); bf2atom = build_bf2atom(td.wfs)
    S_MM = td.wfs.S_qMM[0].copy().real
    rho_MM_eq = dmat.get_density_matrix()[0]
    if rho_MM_eq.shape[1] != S_MM.shape[0]:
        rho_MM_eq = rho_MM_eq.T
    rhoS_eq = np.dot(rho_MM_eq, S_MM)
    q_eq = np.zeros(len(td.atoms))
    for mu in range(rhoS_eq.shape[0]):
        q_eq[bf2atom[mu]] += rhoS_eq[mu, mu].real

    obs_q = MullikenObserver(td, dmat, bf2atom, S_MM, q_eq,
                             out_prefix=str(outdir / "mulliken"))
    td.attach(obs_q, 1)  # sample every step

    # ---- Propagate ----
    dt_as = args.dt_as
    td.propagate(dt_as, args.nsteps)

    # ---- Finalize ----
    obs_q.finalize()
    td.write(str(outdir / f'td-{args.dir}.gpw'), mode='all')
    print("[DONE] Outputs under:", outdir)

if __name__ == "__main__":
    main()
