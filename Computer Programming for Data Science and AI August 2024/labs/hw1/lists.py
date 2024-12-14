import random
import itertools


if __name__ == '__main__':
    s1 = "Python is cool"
    print(s1[:6])
    print(s1[-4:])
    print(s1[-7:-5])
    
    s2 = "Malkahc"
    print(s2[::-1])
    
    s3 = "dffdfPkyktkhkokn"
    print(s3[5::2])

    l1 = [i for i in range(10)]
    print(l1)
    l2 = [i for i in range(1, 10)]
    print(l2)
    l3 = [i for i in range(-9, 10, 2)]
    print(l3)
    l3.append(10)
    print(l3)
    print(l3.index(-1))
    print(len(l3))
    l3.remove(10)
    print(l3)
    l3.remove(l3.index(5))
    print(l3)
    print(-20 in l3)

    def shuffle_list(lst):
        n = len(lst)
        for i in range(n - 1, 0, -1):
            j = random.randint(0, i)
            lst[i], lst[j] = lst[j], lst[i]

    shuffle_list(l3)
    print(l3)
    l3.sort()
    print(l3)

    for n in l3:
        print(n, end=' ')
    print()

    random_value = l3[random.randint(0, len(l3) - 1)]
    print(random_value)

    paired_permutations = list(itertools.permutations(l3, 2))
    print(paired_permutations)

    l3_plus_one = [n + 1 for n in l3]
    print(l3_plus_one)

    for index, value in enumerate(l3_plus_one):
        print(f"Index: {index}, Value: {value}")
    
    friends = ["St1", "St2", "St3"]
    ids = [101, 102, 103]

    for name, id_num in zip(friends, ids):
        print(f"Name: {name}, ID: {id_num}")

    
    l4 = [1, 1, 2, 2]
    unique_values = set(l4)
    print(unique_values)
    print(len(unique_values))


    two_channel_eeg_signal1 = [8, 9]
    event1 = 1
    two_channel_eeg_signal2 = [3, 3]
    event2 = 2
    two_channel_eeg_signal3 = [2, 3]
    event3 = 2

    some_nested_list = []
    some_nested_list.append([two_channel_eeg_signal1, event1])
    some_nested_list.append([two_channel_eeg_signal2, event2])
    some_nested_list.append([two_channel_eeg_signal3, event3])

    # 11
    # (1) Using a for loop
    eeg_signals = []
    events = []

    for item in some_nested_list:
        eeg_signals.append(item[0])
        events.append(item[1])

    print("EEG signals:", eeg_signals)
    print("Events:", events)

    # (2) Using list comprehension
    eeg_signals = [item[0] for item in some_nested_list]
    events = [item[1] for item in some_nested_list]

    print("EEG signals (list comprehension):", eeg_signals)
    print("Events (list comprehension):", events)


    # 12
    eeg_event_2 = [item[0] for item in some_nested_list if item[1] == 2]
    print("EEG signals with event = 2:", eeg_event_2)
