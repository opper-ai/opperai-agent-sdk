import os
import subprocess
import sys


def run_example(file_path):
    print(f"\n=== Running example: {file_path} ===")
    try:
        # Use sys.executable to ensure the correct python from venv is used
        result = subprocess.run(
            [sys.executable, file_path], check=True, capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Stderr:\n{result.stderr}")
        print(f"=== Example {file_path} completed successfully ===")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Example {file_path} FAILED !!!")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"!!! Error: Python executable not found to run {file_path} !!!")
        return False
    except Exception as e:
        print(f"!!! An unexpected error occurred while running {file_path}: {e} !!!")
        return False


def find_and_run_examples(base_dir="."):
    examples_dir = os.path.join(base_dir, "examples")
    if not os.path.isdir(examples_dir):
        print(f"Error: Examples directory not found at {examples_dir}")
        sys.exit(1)

    all_examples = []
    for root, _, files in os.walk(examples_dir):
        for file in files:
            if file.endswith(".py"):
                all_examples.append(os.path.join(root, file))

    all_examples.sort()  # Run in a consistent order

    failed_examples = []
    for example_file in all_examples:
        success = run_example(example_file)
        if not success:
            failed_examples.append(example_file)

    print("\n===========================================")
    print("Example Run Summary:")
    print(f"Total examples found: {len(all_examples)}")
    if not failed_examples:
        print("All examples completed successfully! ✅")
    else:
        print(f"Failed examples: {len(failed_examples)} ❌")
        for fail in failed_examples:
            print(f"  - {fail}")
        sys.exit(1)


if __name__ == "__main__":
    find_and_run_examples(
        os.path.dirname(os.path.abspath(__file__)) + "/.."
    )  # Go up one level from scripts to project root
