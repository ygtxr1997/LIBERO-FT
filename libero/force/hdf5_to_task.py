import os
import re
import json
import h5py

from libero.libero import benchmark, get_libero_path


def inspect_hdf5_metadata(h5_path: str):
    assert os.path.exists(h5_path), f"not found: {h5_path}"
    meta = {}

    with h5py.File(h5_path, "r") as f:
        # 顶层 attrs（通常为空或很少）
        # Top-level attrs (usually empty or few)
        meta["root_attrs"] = {k: f.attrs[k] for k in f.attrs.keys()}

        if "data" not in f:
            raise KeyError("HDF5 missing group: /data")

        g = f["data"]

        # data group attrs：LIBERO/robosuite 通常把关键东西写在这里
        # data group attrs: LIBERO/robosuite usually put key info here
        data_attrs = {}
        for k in g.attrs.keys():
            v = g.attrs[k]
            # h5py 有时返回 bytes
            # h5py sometimes returns bytes
            if isinstance(v, (bytes, bytearray)):
                v = v.decode("utf-8", errors="ignore")
            data_attrs[k] = v

        meta["data_attrs"] = data_attrs

        # 常见字段（不保证每个文件都有，但 LIBERO 官方脚本里会写）
        # Common fields (not guaranteed in every file, but usually written by LIBERO official scripts)
        meta["bddl_file_name"] = data_attrs.get("bddl_file_name", None)
        meta["problem_info_raw"] = data_attrs.get("problem_info", None)

        # problem_info 通常是 json 字符串
        # problem_info is usually a json string
        if meta["problem_info_raw"] is not None:
            try:
                meta["problem_info"] = json.loads(meta["problem_info_raw"])
            except Exception:
                meta["problem_info"] = None

        # Number of demos
        demo_keys = sorted(list(g.keys()))
        meta["num_demos"] = len(demo_keys)
        meta["demo_keys_preview"] = demo_keys[:5]

        # model xml of demo_0 (optional, sometimes exists)
        if len(demo_keys) > 0:
            demo0 = g[demo_keys[0]]
            meta["demo0_attrs"] = {k: demo0.attrs[k] for k in demo0.attrs.keys()}

    return meta


def read_hdf5_env_meta(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        g = f["data"]

        bddl_file = g.attrs["bddl_file_name"]
        if isinstance(bddl_file, (bytes, bytearray)):
            bddl_file = bddl_file.decode("utf-8", errors="ignore")

        env_args_raw = g.attrs.get("env_args", None) or g.attrs.get("env_info", None)
        if env_args_raw is not None and isinstance(env_args_raw, (bytes, bytearray)):
            env_args_raw = env_args_raw.decode("utf-8", errors="ignore")

        env_meta = json.loads(env_args_raw) if env_args_raw else {}
        env_kwargs = env_meta.get("env_kwargs", {}) if isinstance(env_meta, dict) else {}

    return bddl_file, env_kwargs


def guess_suite_name_from_bddl(bddl_file_name: str, suite_candidates=None):
    """
    通过 bddl 文件名在各个 suite 里查找，返回命中的 suite_name。
    According to the bddl filename, search through the suites to find and return the matching suite_name.
    """
    if suite_candidates is None:
        suite_candidates = [
            "libero_10",
            "libero_90",
            "libero_spatial",
            "libero_object",
            "libero_goal",
        ]

    target = os.path.basename(bddl_file_name)

    bench = benchmark.get_benchmark_dict()
    for suite_name in suite_candidates:
        if suite_name not in bench:
            continue
        suite = bench[suite_name]()

        for task_id in range(len(suite.tasks)):
            task = suite.get_task(task_id)
            if os.path.basename(task.bddl_file) == target:
                return suite_name

    return None


def norm(s: str):
    s = s.lower()
    s = s.replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s


def find_task_id_by_bddl(task_suite, bddl_file_name: str):
    """
    task_suite: benchmark_dict[suite_name]()
    bddl_file_name: 从 hdf5 读出来的路径 (可能是绝对路径/相对路径)
    bddl_file_name: the path read from hdf5 (could be absolute/relative path)
    """
    target = os.path.basename(bddl_file_name)

    for task_id in range(len(task_suite.tasks)):
        task = task_suite.get_task(task_id)
        if os.path.basename(task.bddl_file) == target:
            return task_id, task

    raise RuntimeError(f"Cannot find task_id by bddl basename={target} in this task_suite")


def get_task_from_hdf5(h5_path: str, default_suite="libero_10"):
    bddl_file, env_kwargs = read_hdf5_env_meta(h5_path)

    suite_name = guess_suite_name_from_bddl(bddl_file) or default_suite
    suite = benchmark.get_benchmark_dict()[suite_name]()

    task_id, task = find_task_id_by_bddl(suite, bddl_file)

    return {
        "suite_name": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "bddl_file_from_h5": bddl_file,
        "env_kwargs": env_kwargs,
    }
