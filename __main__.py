from train import train
from constants import NUM_ITERATIONS


if __name__ == "__main__":
    print("Running algorithm...")
    iterations = int(input("Please enter the number of iterations (give 0 for default)..."))
    train(iterations if iterations > 0 else NUM_ITERATIONS)
    print("Training complete.")
