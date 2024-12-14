from time import sleep


def accept_dict(**dict):
    print(f"{dict['server']}:{dict['port']}")


def max(nums):
    max_num = -float('inf')

    for num in nums:
        if num > max_num:
            num = max_num
    return num


def countdown():
    counts = 3
    while counts:
        print(counts)
        sleep(1)
        counts -= 1
    print("done!")


if __name__ == '__main__':
    numbers = [6,1,2,7,8,10,2,3]

    # dict = {"server":"localhost", "port":3306, "user":"root", "password":"anything"}
    accept_dict(server="localhost", port=3306, user="root", password="anything")

    print(max(numbers))
    countdown()
