from train import RunConfig, run_training


def main():
    lambda_values = [0.0001, 0.001, 0.01]

    for lambda_reg in lambda_values:
        print("=" * 80)
        print(f"running experiment with lambda_reg={lambda_reg}")
        config = RunConfig(lambda_reg=lambda_reg)
        run_training(config)


if __name__ == "__main__":
    main()
