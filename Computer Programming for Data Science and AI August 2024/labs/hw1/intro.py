from random import randint

if __name__ == '__main__':
    user_input = int(input())

    if user_input < 5:
        print(f"{user_input} is less than 5")
    elif user_input > 5:
        print(f"{user_input} is greater than 5")
    else:
        print(f"{user_input} is equal 5")

    random_n = randint(0, 101)
    
    user_input = -1
    while random_n != user_input:
        user_input = int(input())
        
        if user_input < random_n:
            print(f"User input: {user_input} is less than the guessed number")
        elif user_input > random_n:
            print(f"User input: {user_input} is greater than the guessed number")
        else:
            print(f"Congratulations, you found the guessed number!")
