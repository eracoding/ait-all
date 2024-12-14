def f1(age, *args, name="N/A", **kwargs):
    total_salary = sum(args)
    
    print(f"{name} has an age of {age} with a family salary of {total_salary}")
    
    for kid, school in kwargs.items():
        print(f"The school of {kid} is {school}")


if __name__ == '__main__':
    f1(30, 30000, 40000, name='Chaky', John="some_school", Peter="another_school")

    data = [
            ('Tom', 19, 80),
            ('John', 20, 90),
            ('Jony', 17, 91),
            ('Jony', 17, 93),
            ('Json', 21, 85),
    ]

    sorted_data = sorted(data, key=lambda x: (x[0], x[1], x[2]))

    print(sorted_data)
