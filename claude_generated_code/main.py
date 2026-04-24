def calculate_pi(num_digits: int = 5) -> float:
    """
    Calculate pi using the Leibniz formula for pi:
    pi/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...

    Args:
        num_digits: Number of decimal places to round to (default: 5)

    Returns:
        The value of pi rounded to num_digits decimal places.
    """
    pi_over_4 = 0.0
    num_iterations = 1_000_000  # More iterations = more precision
    for i in range(num_iterations):
        pi_over_4 += ((-1) ** i) / (2 * i + 1)

    pi = 4 * pi_over_4
    return round(pi, num_digits)


if __name__ == "__main__":
    pi = calculate_pi()
    print(f"Pi to the 5th digit: {pi}")
