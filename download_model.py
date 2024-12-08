import os

import gdown


def main():
    gdown.download(id="1BH3Wb9mQhT0mRK1iV1tA3gP5hQMUQRFs")
    os.makedirs("data/main_models/", exist_ok=True)
    os.rename("model_best.pth", "data/main_models/model_best.pth")


if __name__ == "__main__":
    main()
