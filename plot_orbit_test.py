#!/usr/bin/env python3
"""
Plot the position and velocity history of the single test particle from a
tristan-mp-pu user_orbit_test.F90 run (single_particle_test = 1), by reading
the ion datasets directly out of the prtl.tot.* HDF5 files -- no Iseult app
needed, just h5py, for a quick sanity check of the orbit.

Usage:
    python3 plot_orbit_test.py --dir /path/to/output --mx0 260 --my0 260

If --mx0/--my0 are given, positions are also re-centred on the assumed box
centre (mx0/2+0.5, my0/2+0.5) so it can be compared against the ellipse-fit
centre estimated directly from the trajectory (see below); if omitted, only
the trajectory-based estimate is shown.

WHY TWO DIFFERENT "CENTRES":
A naive r(t) = |pos - assumed_box_centre| is only as good as that assumed
centre. If it's off by even a fraction of a cell (e.g. a stale +0.5
convention, or wrong mx0/my0), a perfectly circular orbit will show up as
"elliptical" -- that's a measurement artifact, not real physics. This script
also estimates the centre straight from the (x, y) trajectory, with no
assumption about the box at all, via a proper direct least-squares conic fit
(Fitzgibbon et al. 1996) to the (x, y) points, which recovers the ellipse's
true centre, semi-major/minor axes and eccentricity without assuming *any*
centre up front. This is the most trustworthy of the two, provided the data
covers a decent angular arc of the orbit (a short arc is a genuinely
under-determined fit -- the script reports the angular coverage and warns
when the fit is likely unreliable).

If the assumed box centre disagrees with the fitted centre by more than a
fraction of a cell, that's a strong sign the "eccentricity" you're seeing is
(at least partly) a centring bug rather than real orbital physics.

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


def fit_ellipse(x, y):
    """Direct least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1996).

    Fits the general conic Ax^2+Bxy+Cy^2+Dx+Ey+F=0 under the constraint
    4AC-B^2=1, which guarantees an ellipse-specific solution (never a
    hyperbola/parabola) via a generalized eigenvalue problem. Returns a dict
    with the fitted centre, semi-major axis `a`, semi-minor axis `b`,
    rotation `angle` (radians), eccentricity `ecc`, and a sampled `curve`
    (x, y) for plotting -- or None if the fit is degenerate (too few points,
    collinear points, or a numerically singular conic).

    Coordinates are isotropically normalised (same scale factor for x and y)
    before fitting, purely for numerical conditioning -- an isotropic
    rescale is a similarity transform, so centre/axes/angle/eccentricity map
    back to original units exactly, unlike an anisotropic (sx != sy) rescale
    which would mix the axes under rotation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 6:
        return None

    mx, my = x.mean(), y.mean()
    s = max(np.hypot(x - mx, y - my).std(), 1e-12)
    xn, yn = (x - mx) / s, (y - my) / s

    D1 = np.column_stack([xn**2, xn * yn, yn**2])
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        C = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
        eigval, eigvec = np.linalg.eig(np.linalg.inv(C) @ M)
    except np.linalg.LinAlgError:
        return None

    cond = 4 * eigvec[0, :] * eigvec[2, :] - eigvec[1, :] ** 2
    valid = np.where(cond > 0)[0]
    if valid.size == 0:
        return None
    a1 = eigvec[:, valid[0]].real
    a2 = T @ a1
    A, B, Cc, D, E, F = a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]

    M0 = np.array([[A, B / 2, D / 2], [B / 2, Cc, E / 2], [D / 2, E / 2, F]])
    M2 = np.array([[A, B / 2], [B / 2, Cc]])
    detM2 = np.linalg.det(M2)
    if abs(detM2) < 1e-14:
        return None

    center_n = np.linalg.solve(M2, [-D / 2, -E / 2])
    eigval2, eigvec2 = np.linalg.eigh(M2)
    detM0 = np.linalg.det(M0)
    if detM0 == 0 or np.any(eigval2 == 0):
        return None
    axes2 = -detM0 / (detM2 * eigval2)
    if np.any(axes2 <= 0):
        return None
    axes_n = np.sqrt(axes2)

    order = np.argsort(axes_n)[::-1]  # descending: major axis first
    a_n, b_n = axes_n[order]
    major_dir_n = eigvec2[:, order[0]]
    angle = np.arctan2(major_dir_n[1], major_dir_n[0])

    xc, yc = center_n[0] * s + mx, center_n[1] * s + my
    a_len, b_len = a_n * s, b_n * s
    ecc = np.sqrt(max(0.0, 1 - (b_len / a_len) ** 2)) if a_len > 0 else np.nan

    theta = np.linspace(0, 2 * np.pi, 360)
    ex = xc + a_len * np.cos(theta) * np.cos(angle) - b_len * np.sin(theta) * np.sin(angle)
    ey = yc + a_len * np.cos(theta) * np.sin(angle) + b_len * np.sin(theta) * np.cos(angle)

    return dict(center=(xc, yc), a=a_len, b=b_len, angle=angle, ecc=ecc, curve=(ex, ey))


def angular_coverage_deg(x, y, cx, cy):
    """How many degrees of azimuth around (cx, cy) the trajectory spans.

    Used to flag ellipse fits that are based on too short an arc to be
    trustworthy -- a short arc is consistent with a wide range of different
    ellipses, so the fit can look confident while being numerically
    underdetermined.
    """
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    return float(np.degrees(theta.max() - theta.min()))


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

    have_box = args.mx0 is not None and args.my0 is not None
    centers = {}
    if have_box:
        # +0.5: matches x0_ext/y0_ext in user_orbit_test.F90 -- the blob density
        # is built on the FFT grid (global cells [3, mx0-2]), whose true centre
        # is mx0/2+0.5, not mx0/2. Must stay in sync with that file's centring.
        centers['assumed box centre'] = (args.mx0 / 2.0 + 0.5, args.my0 / 2.0 + 0.5)

    ell = fit_ellipse(x, y)
    if ell is not None:
        centers['ellipse fit'] = ell['center']

    coverage_ref = centers.get('assumed box centre',
                                centers.get('ellipse fit', (float(x.mean()), float(y.mean()))))
    coverage = angular_coverage_deg(x, y, *coverage_ref)

    eccs = {}
    r_curves = {}
    for name, (cx, cy) in centers.items():
        r = np.hypot(x - cx, y - cy)
        r_curves[name] = r
        if name == 'ellipse fit':
            # (r.max()-r.min())/(r.max()+r.min()) is only == eccentricity when r is
            # measured from a focus. The ellipse-fit centre is the geometric centre,
            # not a focus, so use the axis-ratio formula from the fit itself instead.
            eccs[name] = ell['ecc']
        else:
            eccs[name] = (r.max() - r.min()) / (r.max() + r.min())

    if have_box and ell is not None:
        dx = ell['center'][0] - centers['assumed box centre'][0]
        dy = ell['center'][1] - centers['assumed box centre'][1]
        center_offset = float(np.hypot(dx, dy))
    else:
        center_offset = None

    print(f"trajectory angular coverage: {coverage:.1f} deg")
    for name, (cx, cy) in centers.items():
        print(f"  {name:24s}: centre=({cx:8.3f}, {cy:8.3f})   ecc = {eccs[name]:.4f}")
    if ell is not None:
        print(f"  ellipse fit: a={ell['a']:.4f}  b={ell['b']:.4f}  "
              f"b/a={ell['b']/ell['a']:.4f}  angle={np.degrees(ell['angle']) % 180:.1f} deg")
    else:
        print("  ellipse fit: FAILED (too few/collinear points)")

    ncols = 3
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

    ax = axes[0, 0]
    sc = ax.scatter(x, y, c=lap, cmap='viridis', s=10, zorder=3)
    ax.plot(x, y, lw=0.5, alpha=0.5, color='k')
    markers = {'assumed box centre': ('r', '+'), 'ellipse fit': ('g', '*')}
    for name, (cx, cy) in centers.items():
        color, marker = markers[name]
        ax.scatter([cx], [cy], marker=marker, color=color, s=120, label=name, zorder=4)
    if ell is not None:
        ax.plot(*ell['curve'], color='g', lw=1, ls='--', alpha=0.8, label='ellipse fit', zorder=2)
    ax.legend(fontsize=8)
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

    ax = axes[0, 2]
    for name, r in r_curves.items():
        color, _ = markers[name]
        ax.plot(lap, r, color=color, label=name)
    ax.set_xlabel('time [timesteps]')
    ax.set_ylabel('r [cells]')
    ax.legend(fontsize=8)
    ax.set_title('Radius vs time, per centre estimate\n(flat = circular, oscillating = elliptical)')

    ax = axes[1, 2]
    ax.axis('off')
    lines = [f"frames: {len(lap)}   trajectory angular coverage: {coverage:.1f} deg"]
    if coverage < 300:
        lines.append("  ^ WARNING: less than a full orbit -- the ellipse fit is")
        lines.append("    unreliable/underdetermined below ~300 deg.")
    lines.append("")
    for name, (cx, cy) in centers.items():
        lines.append(f"{name:24s}: ({cx:8.3f}, {cy:8.3f})   ecc = {eccs[name]:.4f}")
    if center_offset is not None:
        lines.append("")
        lines.append(f"ellipse-fit centre vs assumed box centre: offset = {center_offset:.3f} cells")
        if center_offset > 0.1:
            lines.append("  ^ this alone is enough to fake a non-trivial eccentricity --")
            lines.append("    check mx0/my0 and the +0.5 centring convention match the run.")
    if ell is not None:
        lines.append("")
        lines.append(f"ellipse fit: a={ell['a']:.3f}  b={ell['b']:.3f}  "
                      f"angle={np.degrees(ell['angle']) % 180:.1f} deg")
    else:
        lines.append("")
        lines.append("ellipse fit: FAILED (too few/collinear points)")
    lines.append("")
    lines.append(f"speed: min={speed.min():.4g}, max={speed.max():.4g}")
    lines.append(f"speed variation: {(speed.max()-speed.min())/speed.mean()*100:.2f}% of mean")
    ax.text(0.0, 0.5, "\n".join(lines), va='center', fontsize=10, family='monospace')

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved to {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
