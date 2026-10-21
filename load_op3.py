import argparse
import numpy as np
import mujoco
import mujoco.viewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--desired-state",
        action="store_true",
        help="Load the model using the desired joint angles",
    )
    args = parser.parse_args()

    xml_path = "./frasa_env/mujoco_simulator/model/scene.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    if args.desired_state:
        # Desired state in degrees
        desired_state = {
            "elbow": -49.5,
            "shoulder_pitch": 19.5,
            "hip_pitch": -52.0,
            "knee": 79.0,
            "ankle_pitch": -36.5,
            "trunk_pitch": 8.5,
        }

        for name, angle_deg in desired_state.items():
            if name == "trunk_pitch":
                continue

            for side in ("left", "right"):
                joint_name = f"{side}_{name}"

                joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )

                if joint_id == -1:
                    raise ValueError(
                        f"Unable to find joint: '{joint_name}'"
                    )

                qpos_addr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_addr] = np.deg2rad(angle_deg)

        # Apply trunk pitch to the free root joint
        trunk_pitch = np.deg2rad(desired_state["trunk_pitch"])
        data.qpos[3:7] = [
            np.cos(trunk_pitch / 2),
            0.0,
            np.sin(trunk_pitch / 2),
            0.0,
        ]

        mujoco.mj_forward(model, data)
        print("Loaded model with desired state.")
    else:
        print("Loaded model with its default initial state.")

    # Display without advancing physics
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if not args.desired_state:
                mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
