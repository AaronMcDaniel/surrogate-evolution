def print_data_file():
    try:
        # Open the file in binary mode (use "r" mode if it's a text file)
        with open("data.pt", "rb") as file:
            content = file.read()
            print(content)
    except FileNotFoundError:
        print("File data.pt not found.")
    except Exception as e:
        print("An error occurred:", e)

# Call the function
print_data_file()
