from typing import Optional
from calculator import Calculator
from user_interface import UserInterface

def main():
    ui = UserInterface()
    calc = Calculator()

    while True:
        operation = ui.get_input()
        if not operation:
            break

        if operation == 'exit':
            break

        try:
            a, b = map(float, operation.split())
        except ValueError:
            ui.display("Invalid input. Please enter numbers separated by space.")
            continue

        if operation == '+':
            result = calc.add(a, b)
        elif operation == '-':
            result = calc.subtract(a, b)
        elif operation == '*':
            result = calc.multiply(a, b)
        elif operation == '/':
            try:
                result = calc.divide(a, b)
            except ZeroDivisionError:
                ui.display("Cannot divide by zero.")
                continue
        else:
            ui.display("Invalid operation. Please choose +, -, *, or /.")
            continue

        ui.display(f"Result: {result}")

if __name__ == "__main__":
    main()
