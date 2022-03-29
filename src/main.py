from src import feature_engineering, classification


def main():
    processed_df = feature_engineering.preprocess_data()
    classification.classify(processed_df)


if __name__ == "__main__":
    main()
