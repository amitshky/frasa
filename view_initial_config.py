import pickle
import random
import time

import numpy as np
import mujoco
import mujoco.viewer


def main():
    xml_path = "./frasa_env/mujoco_simulator/model/scene.xml"
    pkl_path = "./frasa_env/env/standup_initial_configurations.pkl"

    num_configs = 10
    display_seconds = 2.0

    # Load saved configurations
    with open(pkl_path, "rb") as f:
        configs = pickle.load(f)

    if not configs:
        raise ValueError("No configurations found in the pickle file.")

    # Select a few configurations randomly, without repetition
    selected_indices = random.sample(
        range(len(configs)),
        min(num_configs, len(configs)),
    )

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for index in selected_indices:
            if not viewer.is_running():
                break

            qpos, ctrl = configs[index]
            qpos = np.asarray(qpos)
            ctrl = np.asarray(ctrl)

            if qpos.size != model.nq or ctrl.size != model.nu:
                raise ValueError(
                    f"Config {index} dimensions don't match the model: "
                    f"qpos {qpos.size}/{model.nq}, "
                    f"ctrl {ctrl.size}/{model.nu}"
                )

            # Load this configuration
            data.qpos[:] = qpos
            data.ctrl[:] = ctrl
            data.qvel[:] = 0

            # Update derived positions and orientations
            mujoco.mj_forward(model, data)
            viewer.sync()

            print(
                f"Showing config {index} "
                f"({selected_indices.index(index) + 1}/"
                f"{len(selected_indices)})"
            )

            # Hold this pose in the viewer for 2 seconds
            start = time.monotonic()
            while (viewer.is_running()):
                viewer.sync()
                if time.monotonic() - start >= display_seconds:
                    break
                time.sleep(0.01)

    print("Finished displaying configurations.")


if __name__ == "__main__":
    main()
