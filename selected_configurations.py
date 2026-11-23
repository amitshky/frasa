import pickle

# Input and output files
input_pkl = "./frasa_env/env/standup_initial_configurations_op3.pkl"
output_pkl = "./frasa_env/env/standup_selected_configurations.pkl"

# Configuration indices to keep
selected_indices = [
    25, 34, 54, 55, 56, 58, 59, 70, 78,
    105, 124, 125, 174, 167, 177, 183, 199, 213
]


def main():
    # Load original configurations
    with open(input_pkl, "rb") as f:
        configs = pickle.load(f)

    print(f"Total configurations in original file: {len(configs)}")

    # Check that all requested indices exist
    invalid_indices = [
        i for i in selected_indices
        if i < 0 or i >= len(configs)
    ]

    if invalid_indices:
        raise IndexError(
            f"These indices do not exist in the input file: {invalid_indices}"
        )

    # Select configurations in the specified order
    selected_configs = [configs[i] for i in selected_indices]

    # Save selected configurations
    with open(output_pkl, "wb") as f:
        pickle.dump(selected_configs, f)

    print(f"Selected configurations: {len(selected_configs)}")
    print(f"Saved to: {output_pkl}")


if __name__ == "__main__":
    main()
