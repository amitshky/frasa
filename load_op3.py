import numpy as np
import mujoco
import mujoco.viewer


def main():
    xml_path = "./frasa_env/mujoco_simulator/model/scene.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Desired state in degrees
    desired_state = {
        "elbow": -49.5,
        "shoulder_pitch": -19.5,
        "hip_pitch": -52.0,
        "knee": 79.0,
        "ankle_pitch": -36.5,
        "trunk_pitch": 8.5,
    }

    # Set all joint angles (convert degrees to radians)
    for name, angle_deg in desired_state.items():
        if name == "trunk_pitch":
            continue

        right_joint = f"right_{name}"
        left_joint = f"left_{name}"

        joint_id_right = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, right_joint
        )
        joint_id_left = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, left_joint
        )

        if joint_id_right == -1:
            raise ValueError(f"Unable to find joint: {right_joint}")

        if joint_id_left == -1:
            raise ValueError(f"Unable to find joint: {left_joint}")

        qpos_addr = model.jnt_qposadr[joint_id_right]
        data.qpos[qpos_addr] = np.deg2rad(angle_deg)

        qpos_addr = model.jnt_qposadr[joint_id_left]
        data.qpos[qpos_addr] = np.deg2rad(angle_deg)

    # Apply trunk pitch to the free root joint
    trunk_pitch = np.deg2rad(desired_state["trunk_pitch"])
    data.qpos[3:7] = [
        np.cos(trunk_pitch / 2),
        0.0,
        np.sin(trunk_pitch / 2),
        0.0,
    ]

    # Update the model without advancing physics
    mujoco.mj_forward(model, data)

    # Display the pose without running physics
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
