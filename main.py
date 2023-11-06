# This is a sample Python script.

# Press Skift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from TNM112_lab1.keras_mlp import KerasMLP
from TNM112_lab1.data_generator import synthetic_data


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

data = synthetic_data(5)
model = KerasMLP(data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
