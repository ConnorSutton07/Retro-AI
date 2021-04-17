from train import run
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retro AI")
    parser.add_argument("--episodes", type=int, nargs='?',
                        const=True, default=1000, 
                        help="Number of training episodes")
    args = parser.parse_args()
    run(training_mode=False, pretrained=True, num_episodes=300)