import argparse
from modules.ela import run_ela
from modules.metadata import extract_metadata
from modules.noise import run_noise_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    ela_output = "output/ela_edited_result.jpeg"
    noise_output = "output/noise_result.jpeg"

    print("Running ELA...")
    run_ela(args.image_path, ela_output)
    print(f"ELA complete - saved to {ela_output}")

    print("\nRunning Metadata...")
    extract_metadata(args.image_path)
    print("Metadata extraction completed")

    print("\nRunning Noise Analysis...")
    run_noise_analysis(args.image_path, noise_output)
    print(f"Noise analysis complete - saved to {noise_output}")
