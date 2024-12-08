import os

import gdown


def main():
    gdown.download(id="1LoU_kCzl20hM5p709teRPB0a4Jib8VK-")
    os.makedirs("data/main_models/", exist_ok=True)
    os.rename("model_best.pth", "data/main_models/model_best.pth")


if __name__ == "__main__":
    main()
