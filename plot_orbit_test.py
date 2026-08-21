#!/usr/bin/env python3
"""
Plot the position and velocity history of the single test particle from a
tristan-mp-pu user_orbit_test.F90 run (single_particle_test = 1), by reading
the ion datasets directly out of the prtl.tot.* HDF5 files -- no Iseult app
needed, just h5py, for a quick sanity check of the orbit.

Usage:
    python3 plot_orbit_test.py --dir /path/to/output --mx0 260 --my0 260

If --mx0/--my0 are given, positions are re-centred on the box centre
(mx0/2, my0/2) so the orbital radius r(t) can be plotted; if omitted, only
the raw trajectory and velocity components are shown.

Assumes exactly one ion (the test particle) and zero electrons per frame,
which is what single_particle_test = 1 produces.
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt


def natural_index(path):
    m = re.search(r'\.(\d+)$', path)
    return int(m.group(1)) if m else -1


def load_series(files, interval):
    lap, x, y, u, v = [], [], [], [], []
    for f in files:
        ind = natural_index(f)
        with h5py.File(f, 'r') as h5:
            if 'xi' not in h5 or h5['xi'].shape[0] == 0:
                continue  # no ion in this frame (shouldn't happen once the particle exists)
            xi = np.atleast_1d(h5['xi'][()])
            yi = np.atleast_1d(h5['yi'][()])
            ui = np.atleast_1d(h5['ui'][()])
            vi = np.atleast_1d(h5['vi'][()])

        if xi.size > 1 and f is files[0]:
            print(f"warning: {f} has {xi.size} ions, expected 1 -- using index 0. "
                  "(single_particle_test may not be on, or you're looking at a disk run.)")

        lap.append(ind * interval)
        x.append(xi[0])
        y.append(yi[0])
        u.append(ui[0])
        v.append(vi[0])

    order = np.argsort(lap)
    return (np.array(lap)[order], np.array(x)[order], np.array(y)[order],
            np.array(u)[order], np.array(v)[order])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='output', help='directory containing prtl.tot.* files')
    ap.add_argument('--interval', type=int, default=1,
                     help='timesteps between output frames ("interval" in the input file); '
                          'used only to label the time axis in simulation timesteps')
    ap.add_argument('--mx0', type=float, default=None, help='global grid size in x (cells)')
    ap.add_argument('--my0', type=float, default=None, help='global grid size in y (cells)')
    ap.add_argument('--out', default=None, help='save figure to this file instead of showing it')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, 'prtl.tot.*')), key=natural_index)
    if not files:
        raise SystemExit(f"No prtl.tot.* files found in {args.dir!r}")

    lap, x, y, u, v = load_series(files, args.interval)
    if len(lap) == 0:
        raise SystemExit("No ion found in any frame -- check single_particle_test=1 and weight_test.")

    speed = np.sqrt(u**2 + v**2)

    have_center = args.mx0 is not None and args.my0 is not None
    if have_center:
        # +0.5: matches x0_ext/y0_ext in user_orbit_test.F90 -- the blob density
        # is built on the FFT grid (global cells [3, mx0-2]), whose true centre
        # is mx0/2+0.5, not mx0/2. Must stay in sync with that file's centring.
        x0, y0 = args.mx0 / 2.0 + 0.5, args.my0 / 2.0 + 0.5
        r = np.sqrt((x - x0)**2 + (y - y0)**2)

    ncols = 3 if have_center else 2
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=lap, cmap='viridis', s=10)
    ax.plot(x, y, lw=0.5, alpha=0.5, color='k')
    if have_center:
        ax.scatter([x0], [y0], marker='+', color='r', s=100, label='box centre')
        ax.legend()
    ax.set_xlabel('x [cells]')
    ax.set_ylabel('y [cells]')
    ax.set_title('Trajectory (colour = time)')
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='time [timesteps]')

    ax = axes[0, 1]
    ax.plot(lap, x, label='x')
    ax.plot(lap, y, label='y')
    ax.set_xlabel('time [timesteps]')
    ax.set_ylabel('position [cells]')
    ax.legend()
    ax.set_title('Position vs time')

    ax = axes[1, 0]
    ax.plot(lap, u, label='u')
    ax.plot(lap, v, label='v')
    ax.set_xlabel('time [timesteps]')
    ax.set_ylabel('momentum [gamma*beta]')
    ax.legend()
    ax.set_title('Velocity components vs time')

    ax = axes[1, 1]
    ax.plot(lap, speed)
    ax.set_xlabel('time [timesteps]')
    ax.set_ylabel('|u,v| [gamma*beta]')
    ax.set_title('Speed vs time (should stay ~constant\nfor a gravity-only orbit)')

    if have_center:
        ax = axes[0, 2]
        ax.plot(lap, r)
        ax.set_xlabel('time [timesteps]')
        ax.set_ylabel('r [cells]')
        ax.set_title('Radius vs time\n(flat = circular, oscillating = elliptical)')

        ax = axes[1, 2]
        ax.axis('off')
        text = (f"frames: {len(lap)}\n"
                f"r_min = {r.min():.2f} cells\n"
                f"r_max = {r.max():.2f} cells\n"
                f"(r_max-r_min)/(r_max+r_min) ~ eccentricity = {(r.max()-r.min())/(r.max()+r.min()):.3f}\n"
                f"speed: min={speed.min():.4g}, max={speed.max():.4g}\n"
                f"speed variation: {(speed.max()-speed.min())/speed.mean()*100:.2f}% of mean")
        ax.text(0.0, 0.5, text, va='center', fontsize=11, family='monospace')

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved to {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
