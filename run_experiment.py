import subprocess
import sys


if len(sys.argv) != 2:
    print("Usage: python run_experiment.py <number_of_strategies>")
    sys.exit(1)


try:
    N = int(sys.argv[1])
except ValueError:
    print("Number of strategies must be an integer.")
    sys.exit(1)


if N <= 0:
    print("Number of strategies must be greater than 0.")
    sys.exit(1)


for i in range(N):

    print(f"\n{'=' * 50}")
    print(f"STRATEGY {i + 1}/{N}")
    print(f"{'=' * 50}")

    result = subprocess.run(
        ["python", "generate_strategy.py"]
    )

    if result.returncode != 0:
        print(
            f"\nStrategy {i + 1} failed during generation/evaluation."
        )
        print("Continuing to next strategy...")