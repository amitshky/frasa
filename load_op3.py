import mujoco
import mujoco.viewer

# Path to your OP3 XML file
xml_path = "./frasa_env/mujoco_simulator/model/scene.xml"

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(xml_path)

# Create simulation data
data = mujoco.MjData(model)

# Open the MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
