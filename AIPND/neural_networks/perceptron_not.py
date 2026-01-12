# TODO: Set not_weight1, not_weight2, and not_bias
not_weight1 = -1
not_weight2 = -3
not_bias = 2

### Notebook grading

# Inputs and outputs (True only if the second value is 0)
not_test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
not_correct_outputs = [True, False, True, False]
not_outputs = []

# Generate output
for test_input in not_test_inputs:
    linear_combination = not_weight1 * test_input[0] \
                         + not_weight2 * test_input[1] \
                         + not_bias
    output = linear_combination >= 0
    not_outputs.append(output)

print(not_weight1 * 0 + not_bias, not_weight2 * 0 + not_bias)
print(not_weight1 * 0 + not_bias, not_weight2 * 1 + not_bias)
print(not_weight1 * 1 + not_bias, not_weight2 * 0 + not_bias)
print(not_weight1 * 1 + not_bias, not_weight2 * 1 + not_bias)
print(not_outputs)

# Check output correctness
if not_outputs == not_correct_outputs:
    print('Nice!  You got it all correct.')
else:
    for index in range(len(not_outputs)):
        if not_outputs[index] != not_correct_outputs[index]:
            print("For the input {} your weights and bias \
produced an output of {}. The correct output is {}.".format(
                not_test_inputs[index],
                not_outputs[index],
                not_correct_outputs[index]
            ))
            break