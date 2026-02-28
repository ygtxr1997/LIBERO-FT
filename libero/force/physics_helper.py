from typing import List, Tuple, Any, Union, Dict
import numpy as np

from robosuite.utils.binding_utils import MjModel

from robokit.debug_utils.printer import print_batch


class PhysicsHelper:
    def __init__(self, sim_model: MjModel,
                 object_keywords: Tuple[str, ...],
                 random_seed: int = 42,
                 ):
        self.model = sim_model
        self.object_keywords = object_keywords
        self.random_seed = random_seed

        # 0) List all joints and geoms
        self.all_joints = self.list_all_joints(self.model)
        self.all_geoms = self.list_all_geoms(self.model)
        print(f"[DEBUG] dof_frictionloss: {self.model.dof_frictionloss.shape}, "
              f"dof_damping: {self.model.dof_damping.shape}, "
              f"geom_friction: {self.model.geom_friction.shape}")

        # 1) Find object-related joints and geoms
        self.object_joints = self.find_indices_by_keywords(
            self.model, "joint", self.object_keywords)  # [(id, name), ...]
        self.object_geoms = self.find_indices_by_keywords(
            self.model, "geom", self.object_keywords)  # [(id, name), ...]

        # 2) Snapshot original params
        self.original_params_snap = self.snapshot_params(self.model)
        print_batch("[DEBUG] Original physics params:", self.original_params_snap)

    def restore_original_params(self, model: MjModel = None):
        model = self.model if model is None else model
        self.restore_params(model, self.original_params_snap)
        print("[Info] Restored original physics params.")

    def apply_dynamics_shift(
            self,
            model: MjModel = None,
            joint_ids: List[int] = None,
            geom_ids: List[int] = None,
            rng: np.random.RandomState = None,
            frictionloss_scale: Tuple[float, float] = (0.5, 2.0),
            damping_scale: Tuple[float, float] = (0.5, 2.0),
            sliding_friction_scale: Tuple[float, float] = (0.7, 1.3),
            gravity_z_range: Tuple[float, float] = (-10.3, -9.3),
            tweak_solref_range: Tuple[float, float] = (0.8, 1.2),
            verbose: bool = False,
    ):
        """
        joint_ids: list[int]  index of dof_frictionloss/damping
        geom_ids:  list[int]  idnex of geom friction/solref
        """
        if rng is None:
            rng = np.random.RandomState(self.random_seed)

        model = self.model if model is None else model
        joint_ids = [j[0] for j in self.object_joints] if joint_ids is None else joint_ids
        geom_ids = [g[0] for g in self.object_geoms] if geom_ids is None else geom_ids
        scale_info = {}

        # print("[DEBUG] joints to edit:", [(j, model.joint(j).name) for j in joint_ids])
        # print("[DEBUG] geoms  to edit:", [(g, model.geom(g).name) for g in geom_ids])

        def to_01_range(x: float) -> float:
            assert x > 0, "Scale factor should be positive"
            if x > 1.0: return 1 / x
            return x

        # 0) gravity
        if hasattr(model, "opt") and hasattr(model.opt, "gravity") and gravity_z_range is not None:
            g = np.array(model.opt.gravity).copy()
            # g[0] = rng.uniform(-gravity_z_range[0] * 0.5, gravity_z_range[1] * 0.5)  # if want to add x,y perturbation
            # g[1] = rng.uniform(-gravity_z_range[0] * 0.5, gravity_z_range[1] * 0.5)  # if want to add x,y perturbation
            g[2] = rng.uniform(*gravity_z_range)  # change z only
            model.opt.gravity[:] = g
            scale_info["gravity"] = g[2]

        # 1) joint dof params
        if joint_ids:
            if hasattr(model, "dof_frictionloss") and frictionloss_scale is not None:
                s = rng.uniform(*frictionloss_scale)
                for j in joint_ids:
                    a, b = self.joint_to_dof_range(model, j)
                    if "gripper" in model.joint(j).name.lower():  # no need to set for gripper
                        pass
                    else:
                        model.dof_frictionloss[a:b] *= s
                scale_info["dof_frictionloss_scale"] = s

            if hasattr(model, "dof_damping") and damping_scale is not None:
                s = rng.uniform(*damping_scale)
                for j in joint_ids:
                    a, b = self.joint_to_dof_range(model, j)
                    if "gripper" in model.joint(j).name.lower():  # no need to set for gripper
                        pass
                    else:
                        model.dof_damping[a:b] *= s
                scale_info["dof_damping_scale"] = s

        # 2) geom friction
        if geom_ids and hasattr(model, "geom_friction") and sliding_friction_scale is not None:
            s = rng.uniform(*sliding_friction_scale)
            # geom_friction[i] = [sliding, torsional, rolling]
            gripper_ids = []
            notgripper_ids = []
            for gid in geom_ids:
                name = model.geom(gid).name.lower()
                if any(k in name for k in ["gripper"]):
                    gripper_ids.append(gid)
                else:
                    notgripper_ids.append(gid)

            if gripper_ids:
                model.geom_friction[gripper_ids, 0] *= to_01_range(s)  # always set less friction for gripper
            if notgripper_ids:
                model.geom_friction[notgripper_ids, 0] *= s

            scale_info["geom_sliding_scale"] = s

        # 3) solref/solimp
        if geom_ids and hasattr(model, "geom_solref") and tweak_solref_range is not None:
            # solref: [timeconst, dampratio], timeconst larger -> harder
            # solimp: [dmax, dmin, width/min], dmax and dmin smaller -> harder
            s = rng.uniform(*tweak_solref_range)  # larger s means harder

            gripper_ids = []
            notgripper_ids = []
            for gid in geom_ids:
                name = model.geom(gid).name.lower()
                if any(k in name for k in ["gripper"]):
                    gripper_ids.append(gid)
                else:
                    notgripper_ids.append(gid)

            model.geom_solref[gripper_ids, 0] *= 1 / s  # to be smaller, now larger s means harder
            model.geom_solimp[gripper_ids, :2] *= s

            scale_info["geom_solref_scale"] = s

        if verbose:
            print(f"[Info] Applied dynamics shift: {scale_info}")


    @staticmethod
    def list_all_joints(model: MjModel):
        out = []
        for i in range(model.njnt):
            out.append((i, model.joint(i).name))
        debug_str = "; ".join([f"{i}:{name}" for i, name in out])
        print(f"[DEBUG] All joints: {debug_str}")
        return out

    @staticmethod
    def list_all_geoms(model: MjModel):
        out = []
        for i in range(model.ngeom):
            out.append((i, model.geom(i).name))
        debug_str = "; ".join([f"{i}:{name}" for i, name in out])
        print(f"[DEBUG] All geoms: {debug_str}")
        return out

    @staticmethod
    def find_indices_by_keywords(model, kind: str, keywords):
        """
        kind: "joint" or "geom"
        keywords: list[str]，只要 name 包含任意 keyword 就匹配
        """
        keywords = [k.lower() for k in keywords]
        out = []
        if kind == "joint":
            for i in range(model.njnt):
                name = (model.joint(i).name or "").lower()
                if any(k in name for k in keywords):
                    out.append((i, model.joint(i).name))
        elif kind == "geom":
            for i in range(model.ngeom):
                name = (model.geom(i).name or "").lower()
                if any(k in name for k in keywords):
                    out.append((i, model.geom(i).name))
        else:
            raise ValueError(kind)
        debug_str = "; ".join([f"{i}:{name}" for i, name in out])
        print(f"[DEBUG] keywords related {kind}: {debug_str}")
        return out

    @staticmethod
    def joint_to_dof_range(model, j):
        start = int(model.jnt_dofadr[j])
        end = int(model.jnt_dofadr[j + 1]) if j + 1 < model.njnt else model.nv
        return start, end

    @staticmethod
    def snapshot_params(model: MjModel):
        snap = {}
        snap["gravity"] = np.array(model.opt.gravity).copy()
        if hasattr(model, "dof_frictionloss"):
            snap["dof_frictionloss"] = np.array(model.dof_frictionloss).copy()
        if hasattr(model, "dof_damping"):
            snap["dof_damping"] = np.array(model.dof_damping).copy()
        if hasattr(model, "geom_friction"):
            snap["geom_friction"] = np.array(model.geom_friction).copy()
        if hasattr(model, "geom_solref"):
            snap["geom_solref"] = np.array(model.geom_solref).copy()
        if hasattr(model, "geom_solimp"):
            snap["geom_solimp"] = np.array(model.geom_solimp).copy()
        return snap

    @staticmethod
    def restore_params(model: MjModel, snap: Dict[str, np.ndarray]):
        model.opt.gravity[:] = snap["gravity"]
        if "dof_frictionloss" in snap:
            model.dof_frictionloss[:] = snap["dof_frictionloss"]
        if "dof_damping" in snap:
            model.dof_damping[:] = snap["dof_damping"]
        if "geom_friction" in snap:
            model.geom_friction[:] = snap["geom_friction"]
        if "geom_solref" in snap:
            model.geom_solref[:] = snap["geom_solref"]
        if "geom_solimp" in snap:
            model.geom_solimp[:] = snap["geom_solimp"]

