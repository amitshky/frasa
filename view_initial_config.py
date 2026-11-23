import pickle
import time
import threading

import numpy as np
import mujoco
import mujoco.viewer


def main():
    xml_path = "./frasa_env/mujoco_simulator/model/scene.xml"
    pkl_path = "./frasa_env/env/standup_initial_configurations.pkl"

    display_seconds = 2.0

    # Load saved configurations
    with open(pkl_path, "rb") as f:
        configs = pickle.load(f)

    if not configs:
        raise ValueError("No configurations found in the pickle file.")

    num_configs = len(configs)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Playback state
    current_index = 0
    paused = False
    running = True
    last_change = time.monotonic()

    # Lock shared state between keyboard callbacks and main loop
    state_lock = threading.Lock()

    def key_callback(keycode):
        nonlocal current_index, paused, running, last_change

        with state_lock:
            if keycode == ord('Q'):
                running = False
                print("Quitting...")

            elif keycode == ord('P'):
                paused = not paused
                last_change = time.monotonic()
                print("Paused" if paused else "Resumed")

            elif keycode == 262:  # Right arrow
                current_index = (current_index + 1) % num_configs
                last_change = time.monotonic()
                print(f"Next: configuration {current_index}")

            elif keycode == 263:  # Left arrow
                current_index = (current_index - 1) % num_configs
                last_change = time.monotonic()
                print(f"Previous: configuration {current_index}")

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
    ) as viewer:

        while viewer.is_running():

            with state_lock:
                if not running:
                    break

                index = current_index
                is_paused = paused

            # Load configuration
            qpos, ctrl = configs[index]
            qpos = np.asarray(qpos)
            ctrl = np.asarray(ctrl)

            if qpos.size != model.nq or ctrl.size != model.nu:
                raise ValueError(
                    f"Config {index} dimensions don't match the model: "
                    f"qpos {qpos.size}/{model.nq}, "
                    f"ctrl {ctrl.size}/{model.nu}"
                )

            data.qpos[:] = qpos
            data.ctrl[:] = ctrl
            data.qvel[:] = 0

            mujoco.mj_forward(model, data)
            viewer.sync()

            print(
                f"Showing config id (starts with 0): {index}/{num_configs - 1}"
                + (" [PAUSED]" if is_paused else "")
            )

            # Display configuration until timeout or manual navigation
            while viewer.is_running():

                with state_lock:
                    if not running:
                        break

                    elapsed = time.monotonic() - last_change
                    is_paused = paused
                    new_index = current_index

                    if not is_paused and elapsed >= display_seconds:
                        current_index = (current_index + 1) % num_configs
                        last_change = time.monotonic()
                        break

                    if new_index != index:
                        break

                viewer.sync()
                time.sleep(0.01)

            with state_lock:
                if not running:
                    break

    print("Finished displaying configurations.")


if __name__ == "__main__":
    main()
