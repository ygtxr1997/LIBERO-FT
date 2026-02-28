import h5py
import numpy as np
import math
from collections import defaultdict

def _psnr_uint8(a, b, eps=1e-12):
    # a,b: uint8 arrays same shape
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < eps:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)

def _diff_numeric(a, b):
    diff = a.astype(np.float64) - b.astype(np.float64)
    absd = np.abs(diff)
    return {
        "max_abs": float(np.max(absd)),
        "mean_abs": float(np.mean(absd)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }

def _iter_datasets(g, prefix=""):
    """Yield (path, dataset) for all datasets under group g."""
    for k, v in g.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Dataset):
            yield p, v
        elif isinstance(v, h5py.Group):
            yield from _iter_datasets(v, p)

def _attrs_to_dict(h5obj):
    out = {}
    for k, v in h5obj.attrs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.generic,)):
            out[k] = v.item()
        else:
            out[k] = v
    return out

def _align_time(ds_a, ds_b, allow_time_len_diff_1=True):
    """
    Return (sa, sb, note) where sa/sb are tuple slices to align arrays for comparison.
    - Exact match: compare full.
    - If allow_time_len_diff_1 and rank>=1 and shapes match except axis0 differs by 1:
        compare prefix min(Ta,Tb) along axis0.
    Otherwise return None.
    """
    if ds_a.dtype != ds_b.dtype:
        return None

    if ds_a.shape == ds_b.shape:
        sa = tuple([slice(None)] * len(ds_a.shape))
        sb = tuple([slice(None)] * len(ds_b.shape))
        return sa, sb, None

    if not allow_time_len_diff_1:
        return None

    if len(ds_a.shape) >= 1 and len(ds_a.shape) == len(ds_b.shape):
        if ds_a.shape[1:] == ds_b.shape[1:] and abs(ds_a.shape[0] - ds_b.shape[0]) == 1:
            minT = min(ds_a.shape[0], ds_b.shape[0])
            sa = [slice(None)] * len(ds_a.shape)
            sb = [slice(None)] * len(ds_b.shape)
            sa[0] = slice(0, minT)
            sb[0] = slice(0, minT)
            note = ("time_len_diff_1", ds_a.shape[0], ds_b.shape[0], minT)
            return tuple(sa), tuple(sb), note

    return None

def compare_hdf5(
    h5_a_path,
    h5_b_path,
    max_demos=None,
    sample_frames=20,
    allow_time_len_diff_1=True,
    only_report_changed=True,
    print_limits=None,
):
    """
    Compare two HDF5 files with structure:
      /data/demo_x/{actions,dones,rewards,robot_states,states,obs/...}

    Features:
    - Compares global/data/demo attrs
    - Compares datasets for:
        * missing/extra
        * dtype/shape mismatch
        * numeric diffs (max_abs/mean_abs/rmse)
        * image diffs for uint8 RGB(A): mean |pixdiff|, diff_ratio, PSNR
    - NEW: Supports time length mismatch of exactly 1 at axis0 (T vs T-1),
          by comparing prefix min(Ta,Tb). Records this under TIME LENGTH DIFF.

    Args:
      sample_frames: number of frames to sample for image datasets; None => all frames
      only_report_changed: if True, numeric_diff only recorded when max_abs > 0
      print_limits: dict to override how many entries to print per section
    """
    if print_limits is None:
        print_limits = {
            "attrs": 50,
            "missing": 30,
            "extra": 30,
            "shape": 50,
            "time_len": 80,
            "numeric": 120,
            "image": 80,
        }

    report = defaultdict(list)

    def diff_attrs(name, oa, ob):
        da, db = _attrs_to_dict(oa), _attrs_to_dict(ob)
        ka, kb = set(da.keys()), set(db.keys())
        only_a, only_b = sorted(ka - kb), sorted(kb - ka)
        common = sorted(ka & kb)
        changed = []
        for k in common:
            if da[k] != db[k]:
                changed.append((k, da[k], db[k]))
        if only_a or only_b or changed:
            report["attrs"].append((name, only_a, only_b, changed))

    with h5py.File(h5_a_path, "r") as fa, h5py.File(h5_b_path, "r") as fb:
        # attrs: global + /data
        diff_attrs("global", fa, fb)
        if "data" in fa and "data" in fb:
            diff_attrs("/data", fa["data"], fb["data"])

        demos_a = sorted(list(fa["data"].keys()))
        demos_b = sorted(list(fb["data"].keys()))
        demos_common = [d for d in demos_a if d in demos_b]
        if max_demos is not None:
            demos_common = demos_common[:max_demos]

        # per-demo
        for demo in demos_common:
            ga, gb = fa["data"][demo], fb["data"][demo]
            diff_attrs(f"/data/{demo}", ga, gb)

            dsa = {p: ds for p, ds in _iter_datasets(ga, prefix=f"data/{demo}")}
            dsb = {p: ds for p, ds in _iter_datasets(gb, prefix=f"data/{demo}")}

            pa, pb = set(dsa.keys()), set(dsb.keys())
            only_a, only_b = sorted(pa - pb), sorted(pb - pa)
            if only_a:
                report["missing_in_new"].append((demo, only_a))
            if only_b:
                report["extra_in_new"].append((demo, only_b))

            for p in sorted(pa & pb):
                da, db = dsa[p], dsb[p]

                align = _align_time(da, db, allow_time_len_diff_1=allow_time_len_diff_1)

                if align is None:
                    report["shape_dtype_mismatch"].append(
                        (demo, p, da.shape, db.shape, str(da.dtype), str(db.dtype))
                    )
                    continue

                sa, sb, note = align
                if note is not None and note[0] == "time_len_diff_1":
                    _, Ta, Tb, minT = note
                    report["time_len_diff"].append((demo, p, Ta, Tb, minT))

                # IMAGE datasets (uint8, rank 4, last dim 3/4)
                if da.dtype == np.uint8 and len(da.shape) == 4 and da.shape[-1] in (3, 4):
                    a_all = da[sa]
                    b_all = db[sb]
                    T = a_all.shape[0]
                    if sample_frames is None:
                        idxs = list(range(T))
                    else:
                        idxs = np.linspace(0, T - 1, min(sample_frames, T), dtype=int).tolist()

                    mad_list, psnr_list, diff_ratio_list = [], [], []
                    for t in idxs:
                        a = a_all[t]
                        b = b_all[t]
                        diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
                        mad_list.append(float(diff.mean()))
                        diff_ratio_list.append(float((diff > 0).mean()))
                        psnr_list.append(_psnr_uint8(a, b))

                    mean_psnr = float(np.mean(psnr_list)) if np.isfinite(np.mean(psnr_list)) else float("inf")
                    # 只报告明显差异（可选）
                    if (not only_report_changed) or (np.mean(mad_list) > 0):
                        report["image_diff"].append(
                            (demo, p, T, len(idxs),
                             float(np.mean(mad_list)),
                             float(np.mean(diff_ratio_list)),
                             mean_psnr)
                        )
                else:
                    # Numeric datasets
                    a = da[sa]
                    b = db[sb]
                    stats = _diff_numeric(a, b)
                    if (not only_report_changed) or (stats["max_abs"] > 0):
                        report["numeric_diff"].append((demo, p, stats))

    # Pretty print
    def print_section(title, items, limit):
        print(f"\n=== {title} (count={len(items)}) ===")
        for it in items[:limit]:
            print(it)
        if len(items) > limit:
            print(f"... ({len(items) - limit} more)")

    print_section("ATTRS DIFF", report["attrs"], print_limits["attrs"])
    print_section("MISSING IN NEW", report["missing_in_new"], print_limits["missing"])
    print_section("EXTRA IN NEW", report["extra_in_new"], print_limits["extra"])
    print_section("SHAPE/DTYPE MISMATCH (not alignable)", report["shape_dtype_mismatch"], print_limits["shape"])
    print_section("TIME LENGTH DIFF (allowed +/-1 on axis0)", report["time_len_diff"], print_limits["time_len"])
    print_section("NUMERIC DIFF", report["numeric_diff"], print_limits["numeric"])

    print("\n=== IMAGE DIFF (mean over sampled frames) ===")
    print("format: (demo, path, aligned_T, sampled, mean|pixdiff|, diff_ratio, mean_psnr)")
    for it in report["image_diff"][:print_limits["image"]]:
        print(it)
    if len(report["image_diff"]) > print_limits["image"]:
        print(f"... ({len(report['image_diff']) - print_limits['image']} more)")

    return report

# -------------------------
# Example usage:
# report = compare_hdf5(
#     "orig.hdf5",
#     "new.hdf5",
#     max_demos=None,
#     sample_frames=20,
#     allow_time_len_diff_1=True,
#     only_report_changed=True,
# )
