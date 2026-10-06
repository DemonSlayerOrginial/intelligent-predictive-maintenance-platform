from src.data.ai4i import download_ai4i


if __name__ == "__main__":
    path = download_ai4i()
    print(f"AI4I dataset saved to {path}")
