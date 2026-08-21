import subprocess
import sys


if len(sys.argv) != 2:
    print("Usage: python run_experiment.py <number_of_strategies>")
    sys.exit(1)

N = int(sys.argv[1])


for i in range(N):
    print(f"\n{'=' * 50}")
    print(f"STRATEGY {i + 1}/{N}")
    print(f"{'=' * 50}")

    subprocess.run(
        ["python", "generate_strategy.py"],
        check=True
    )