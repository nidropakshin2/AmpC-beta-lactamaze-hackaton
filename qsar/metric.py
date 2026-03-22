import joblib


def test_scores(model_file):
    model = joblib.load(f"{model_file}")
    
    return model.metadata['test_scores']



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=False, help='model file (PKL)')
    
    args = parser.parse_args()
    
    print(test_scores(args.model_file))
