import pickle

pkl_path = "/home/frisokroes/loco-mujoco-linux/loco-mujoco/outputs/butterfly-118/PPOJax_saved_butterfly_118.pkl"

with open(pkl_path, 'rb') as f:
    agent_data = pickle.load(f)

# Check the data type and keys
if isinstance(agent_data, dict):
    print("Top-level keys in the pickle file:")
    for key in agent_data.keys():
        print(f" - '{key}'")

    # If there's an 'observations' or 'states' key, let's look at its shape
    for possible_key in ['observations', 'states', 'qpos']:
        if possible_key in agent_data:
            print(f"\nShape of {possible_key}:", type(agent_data[possible_key]))
            try:
                print(agent_data[possible_key].shape)
            except AttributeError:
                pass
else:
    print(f"The pickle file contains a {type(agent_data)}, not a dictionary.")
    # If it's a list, it might be a list of timesteps
    if isinstance(agent_data, list):
        print(f"List length: {len(agent_data)}")
        print("First item type:", type(agent_data[0]))
        if isinstance(agent_data[0], dict):
            print("Keys in the first timestep:", agent_data[0].keys())