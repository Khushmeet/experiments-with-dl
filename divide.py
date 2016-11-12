def divide(suv_value):
    array_1 = [0]
    array_2 = [0]
    array_sum = sum(suv_value)
    each_array_val = array_sum/2;
    suv_value.sort(reverse=True)
    for i in suv_value:
        if (sum(array_1) + i) <= each_array_val:
            array_1.append(i)
    for i in suv_value:
        if i not in array_1:
            if ( sum(array_2) + i ) <= each_array_val:
                array_2.append(i)
    if sum(array_1) == each_array_val and sum(array_2) == each_array_val:
        print('True')
    else:
        print('False')

suv_value = []
val = input("Enter the number of souvenirs");
for i in range(int(val)):
    num = input("enter the value: ");
    suv_value.append(int(num));

divide(suv_value)