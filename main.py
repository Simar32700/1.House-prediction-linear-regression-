"""Entry point for the house price prediction project."""

from predict import HOUSE_FIELDS, load_model, predict_price


def read_number(label, minimum, maximum):
    while True:
        try:
            value = int(input(f"{label}: "))
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < minimum:
            print(f"Please enter {minimum} or more.")
        elif maximum is not None and value > maximum:
            print(f"Please enter {maximum} or less.")
        else:
            return value


def main():
    try:
        model = load_model()
    except FileNotFoundError as error:
        print(error)
        return

    print("House Price Prediction")
    try:
        while True:
            house_details = {}
            for name, label, minimum, maximum in HOUSE_FIELDS:
                house_details[name] = read_number(label, minimum, maximum)
            try:
                price = predict_price(house_details, model)
                print(f"\nEstimated house price: {price:,.2f} (dataset price units)")
            except ValueError as error:
                print(f"\nUnable to predict: {error}")
            if input("\nPredict another house? (y/n): ").strip().lower() != "y":
                break
    except (EOFError, KeyboardInterrupt):
        print("\nPrediction cancelled.")


if __name__ == "__main__":
    main()
