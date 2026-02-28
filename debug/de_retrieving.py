import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero.force.modules import MujocoSensorReader, WrenchObsWrapper
from libero.force.physics_helper import PhysicsHelper

from robokit.debug_utils.printer import print_batch
from robokit.debug_utils.images import (
    plot_action_wrt_time, save_frames_as_video, plot_force_sensor_wrt_time, concatenate_rgb_images
)


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_90" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()  # libero.libero.benchmark.LIBERO_90

## 1) Retrieve a specific task
# for task_id in range(task_suite.get_num_tasks()):
#     task = task_suite.get_task(task_id)
#     print(f"[info] task id: {task_id}, name: {task.name}, language: {task.language}, bddl file: {task.bddl_file}")
# exit()
task_id = 23
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

## 2) Retrieve environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
base_env = OffScreenRenderEnv(**env_args)
env = WrenchObsWrapper(base_env, force_sensor="gripper0_force_ee", torque_sensor="gripper0_torque_ee")

env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

## 3) Physics helper
physics_helper = PhysicsHelper(
    env.env.sim.model,
    object_keywords=("cabinet", "drawer", "door", "basket", "object", "tool")
)
scale_value = 1.
gz_scale_value = 0.1
physics_helper.apply_dynamics_shift(
    frictionloss_scale=(scale_value, scale_value),
    damping_scale=(scale_value, scale_value),
    sliding_friction_scale=(scale_value, scale_value),
    gravity_z_range=(-9.8 * gz_scale_value, -9.8 * gz_scale_value),
)

## 4) Check sensor
print("===" * 20)
print("[DEBUG] Checking sensors in the environment...")
print(type(env))        # OffScreenRenderEnv
print(type(env.env))    # 真实 robosuite env，通常是 BDDLBaseDomain 子类
print(hasattr(env.env, "sim"), hasattr(env.env, "get_sensor_measurement"))

reader = MujocoSensorReader(env.env.sim, ["gripper0_force_ee", "gripper0_torque_ee"])
w = reader.read_wrench("gripper0_force_ee", "gripper0_torque_ee")
print("wrench:", w, w.shape)  # (6,)

print("[DEBUG] got sensors: force_ee, torque_ee")
print("===" * 20)

## 5) Step in the environment and visualize
vis_images = []
vis_forces = []

dummy_action = [0.] * 7
dummy_action[0] = 0.2  # move forward
dummy_action[1] = -0.2  # move left
dummy_action[2] = -0.25  # move down
for step in tqdm(range(300)):
    step_action = [x * 10 for x in dummy_action]
    obs, reward, done, info = env.step(step_action)
    if step == 70:
        dummy_action[0] = -0.2   # moving backward
        dummy_action[1] = 0.7  # move right
        dummy_action[2] = -0.01   # stop moving down
    elif step == 130:
        dummy_action[1] = -0.3  # move left
        dummy_action[2] = -0.2  # move down
    """
    [DEBUG] obs:: Dict, keys=['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'alphabet_soup_1_pos', 'alphabet_soup_1_quat', 'alphabet_soup_1_to_robot0_eef_pos', 'alphabet_soup_1_to_robot0_eef_quat', 'cream_cheese_1_pos', 'cream_cheese_1_quat', 'cream_cheese_1_to_robot0_eef_pos', 'cream_cheese_1_to_robot0_eef_quat', 'tomato_sauce_1_pos', 'tomato_sauce_1_quat', 'tomato_sauce_1_to_robot0_eef_pos', 'tomato_sauce_1_to_robot0_eef_quat', 'ketchup_1_pos', 'ketchup_1_quat', 'ketchup_1_to_robot0_eef_pos', 'ketchup_1_to_robot0_eef_quat', 'orange_juice_1_pos', 'orange_juice_1_quat', 'orange_juice_1_to_robot0_eef_pos', 'orange_juice_1_to_robot0_eef_quat', 'milk_1_pos', 'milk_1_quat', 'milk_1_to_robot0_eef_pos', 'milk_1_to_robot0_eef_quat', 'butter_1_pos', 'butter_1_quat', 'butter_1_to_robot0_eef_pos', 'butter_1_to_robot0_eef_quat', 'basket_1_pos', 'basket_1_quat', 'basket_1_to_robot0_eef_pos', 'basket_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state', 'force_ee', 'torque_ee', 'wrench_ee']
    --robot0_joint_pos, <class 'numpy.ndarray'>, shape=(7,), min=-2.3817, max=2.1704, dtype=float64
    --robot0_joint_pos_cos, <class 'numpy.ndarray'>, shape=(7,), min=-0.7249, max=1.0000, dtype=float64
    --robot0_joint_pos_sin, <class 'numpy.ndarray'>, shape=(7,), min=-0.6888, max=0.8255, dtype=float64
    --robot0_joint_vel, <class 'numpy.ndarray'>, shape=(7,), min=-0.0861, max=0.0721, dtype=float64
    --robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0461, max=0.7020, dtype=float64
    --robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0269, max=0.9996, dtype=float64
    --robot0_gripper_qpos, <class 'numpy.ndarray'>, shape=(2,), min=-0.0205, max=0.0205, dtype=float64
    --robot0_gripper_qvel, <class 'numpy.ndarray'>, shape=(2,), min=-0.0005, max=0.0005, dtype=float64
    --agentview_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=247.0000, dtype=uint8
    --robot0_eye_in_hand_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=198.0000, dtype=uint8
    --alphabet_soup_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.1520, max=0.4752, dtype=float64
    --alphabet_soup_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0024, max=0.7075, dtype=float64
    --alphabet_soup_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0416, max=0.2294, dtype=float64
    --alphabet_soup_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.7065, max=0.7073, dtype=float32
    --cream_cheese_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.1938, max=0.4457, dtype=float64
    --cream_cheese_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=0.0000, max=1.0000, dtype=float64
    --cream_cheese_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=0.1385, max=0.2492, dtype=float64
    --cream_cheese_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.9996, max=0.0269, dtype=float32
    --tomato_sauce_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.1173, max=0.4752, dtype=float64
    --tomato_sauce_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0023, max=0.7075, dtype=float64
    --tomato_sauce_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0592, max=0.2304, dtype=float64
    --tomato_sauce_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.7065, max=0.7073, dtype=float32
    --ketchup_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2685, max=0.5092, dtype=float64
    --ketchup_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=0.5000, max=0.5000, dtype=float64
    --ketchup_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2115, max=0.2045, dtype=float64
    --ketchup_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.5131, max=0.5135, dtype=float32
    --orange_juice_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2592, max=0.5065, dtype=float64
    --orange_juice_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=0.5000, max=0.5000, dtype=float64
    --orange_juice_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=0.0668, max=0.2557, dtype=float64
    --orange_juice_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.5131, max=0.5135, dtype=float32
    --milk_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.1214, max=0.5065, dtype=float64
    --milk_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=0.5000, max=0.5000, dtype=float64
    --milk_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=0.1059, max=0.1901, dtype=float64
    --milk_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.5131, max=0.5135, dtype=float32
    --butter_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=0.0527, max=0.4455, dtype=float64
    --butter_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=0.0000, max=1.0000, dtype=float64
    --butter_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0563, max=0.2499, dtype=float64
    --butter_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.9996, max=0.0269, dtype=float32
    --basket_1_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.0021, max=0.4322, dtype=float64
    --basket_1_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0017, max=0.7071, dtype=float64
    --basket_1_to_robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2725, max=0.2672, dtype=float64
    --basket_1_to_robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.7071, max=0.7066, dtype=float32
    --robot0_proprio-state, <class 'numpy.ndarray'>, shape=(39,), min=-2.3817, max=2.1704, dtype=float64
    --object-state, <class 'numpy.ndarray'>, shape=(112,), min=-0.9996, max=1.0000, dtype=float64
    --force_ee, <class 'numpy.ndarray'>, shape=(3,), min=-5.0862, max=0.0012, dtype=float32
    --torque_ee, <class 'numpy.ndarray'>, shape=(3,), min=-0.0009, max=0.0049, dtype=float32
    --wrench_ee, <class 'numpy.ndarray'>, shape=(6,), min=-5.0862, max=0.0049, dtype=float32
    """
    resized_image = np.array(
        Image.fromarray(obs["agentview_image"]).resize((512, 512))
    )
    vis_images.append(np.flipud(resized_image))
    vis_forces.append(obs["wrench_ee"])

force_frames, _, _ = plot_force_sensor_wrt_time(
    np.array(vis_forces)
)
combined_vis_images = [
    concatenate_rgb_images(img, force_img, resize_ratio=1.)
    for img, force_img in zip(vis_images, force_frames)
]
save_frames_as_video(
    combined_vis_images,
    f"debug/physics_shift_scale{scale_value:.2f}_gz{gz_scale_value:.2f}.mp4",
    fps=10
)

env.close()

