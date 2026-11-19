"""Stabilize per-person world root translation against monocular depth jitter/drift.

Moving-camera monocular reconstruction injects implausible speed/acceleration spikes
into a person's world trajectory (far/small dancers worst). The downstream humanoid
tracking policy cannot follow those and falls. This applies temporal smoothing plus
hard velocity/acceleration caps calibrated to plausible human dance motion, preserving
genuine movement while removing depth noise.
"""
import numpy as np
from scipy.signal import savgol_filter


def detrend_trans(pos, fps, win_seconds=2.0, anchor="first"):
    """Remove low-frequency monocular depth drift, keep local dance motion.

    A far/small dancer's world translation accumulates meter-scale drift from
    unreliable monocular depth; the tracking policy falls trying to follow it.
    Depth drift is low-frequency, genuine dance root motion (steps/bob) is not,
    so subtract a long-window smooth trend and re-anchor. Keeps the dancer near
    its formation slot instead of wandering implausibly.
    """
    p = np.asarray(pos, np.float64).copy()
    T = len(p)
    w = int(round(win_seconds * fps))
    w = min(w, T if T % 2 else T - 1)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return p
    trend = savgol_filter(p, w, 2, axis=0)
    local = p - trend
    base = p[0] if anchor == "first" else p.mean(0)
    return base + local


def needs_stabilize(pos, fps, speed_p95_thresh=3.0, accel_p95_thresh=20.0, skip_frames=60):
    """Flag trajectories whose speed AND acceleration exceed plausible human dance
    motion (signature of monocular depth drift). Early frames are excluded because
    camera/world estimation is unstable there and would over-trigger. Good tracks
    stay untouched."""
    p = np.asarray(pos, np.float64)[skip_frames:]
    if len(p) < 5:
        return False
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
    a = np.linalg.norm(np.diff(p, n=2, axis=0), axis=1) * fps * fps
    return (np.percentile(v, 95) > speed_p95_thresh
            and np.percentile(a, 95) > accel_p95_thresh)


def stabilize_world(pos, fps, win_seconds=2.0):
    """Selective: remove depth drift only for implausible trajectories."""
    if needs_stabilize(pos, fps):
        return detrend_trans(pos, fps, win_seconds=win_seconds)
    return np.asarray(pos, np.float64)


def stabilize_npz(path, win_seconds=2.0):
    """Rewrite a smplx track npz's world 'trans' in place (selective drift removal)."""
    import numpy as _np
    z = _np.load(path, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    trans = data["trans"]
    fps = float(data.get("mocap_frame_rate", 30.0))
    if needs_stabilize(trans, fps):
        data["trans"] = stabilize_world(trans, fps, win_seconds).astype(trans.dtype)
        _np.savez(path, **data)
        return True
    return False
    """pos: (T,3) world root translation. Returns stabilized (T,3).

    vmax m/s, amax m/s^2: hard caps on per-frame speed / acceleration.
    smooth_win: savgol window (frames, odd); <=3 disables smoothing.
    """
    p = np.asarray(pos, np.float64).copy()
    T = len(p)
    if T < 5:
        return p
    if smooth_win >= 5:
        w = min(smooth_win, T if T % 2 else T - 1)
        if w % 2 == 0:
            w -= 1
        if w >= 5:
            p = savgol_filter(p, w, 2, axis=0)
    dt = 1.0 / fps
    max_step = vmax * dt
    max_dv = amax * dt * dt
    for _ in range(iters):
        # forward then backward velocity clamp (removes spikes both directions)
        for direction in (1, -1):
            rng = range(1, T) if direction == 1 else range(T - 2, -1, -1)
            for i in rng:
                j = i - direction
                d = p[i] - p[j]
                n = np.linalg.norm(d)
                if n > max_step:
                    p[i] = p[j] + d * (max_step / n)
        # acceleration clamp on the resulting velocity
        for direction in (1, -1):
            rng = range(2, T) if direction == 1 else range(T - 3, -1, -1)
            for i in rng:
                v1 = p[i] - p[i - direction]
                v0 = p[i - direction] - p[i - 2 * direction]
                dv = v1 - v0
                n = np.linalg.norm(dv)
                if n > max_dv:
                    p[i] = p[i - direction] + v0 + dv * (max_dv / n)
    return p