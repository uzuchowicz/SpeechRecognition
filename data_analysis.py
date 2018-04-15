def second_min(data_array):
    first_min, second_min = float('inf'), float('inf')
    first_min_idx = 0
    second_min_idx =0
    for idx in range(len(data_array)):
        if data_array[idx] <= first_min:
            first_min, second_min= data_array[idx], first_min
            second_min_idx = first_min_idx
            first_min_idx = idx

        elif data_array[idx] < second_min:
            second_min = data_array[idx]
            second_min_idx = idx
    return second_min_idx

x = [1,3,0,2,5,4]
print(second_min(x))
