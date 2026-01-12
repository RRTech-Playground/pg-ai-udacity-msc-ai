# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

################################
## Look at the data

# Reading the csv file into a pandas DataFrame
data = pd.read_csv('student_data.csv')

# Printing out the first 10 rows of our data
print(data[:10])

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')

# Plotting the points
plot_points(data)
#plt.show()

# Separating the ranks
data_rank1 = data[data["rank"]==1]
data_rank2 = data[data["rank"]==2]
data_rank3 = data[data["rank"]==3]
data_rank4 = data[data["rank"]==4]

# Plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
#plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
#plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
#plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
#plt.show()


################################
## One-Hot Encode the Rank

# TODO: Make dummy variables for rank
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)

# TODO: Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)

# Print the first 10 rows of our data
print(one_hot_data[:10])


################################
## Scale the Data

# Making a copy of our data
processed_data = one_hot_data[:]

# TODO: Scale the columns
processed_data['gre'] = processed_data['gre'] / 800
processed_data['gpa'] = processed_data['gpa'] / 4

# Printing the first 10 rows of our procesed data
print(processed_data[:10])


################################
## Split the Data into Training and Test Sets

sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

print("Number of training samples is", len(train_data))
print("Number of test samples is", len(test_data))
print(train_data[:10])
print(test_data[:10])


################################
## Split the Data into Features and Targets

features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])


################################
## Train the 2-Layer Neural Network

# Activation (Sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x, dtype=float)))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# We are using binary cross-entropy (log-loss) as the loss function throughout training,
# including for computing the error term and monitoring progress.
# This ensures consistency with other materials and reflects standard practice
# in classification tasks with sigmoid outputs.
def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1-output)


################################
## Backpropagate the Error

# TODO: Write the error term formula
def error_term_formula(x, y, output):
    return y - output

## alternative solution ##
# you could also *only* use y and the output
# and calculate sigmoid_prime directly from the activated output!

# below is an equally valid solution (it doesn't utilize x)
#def error_term_formula(x, y, output):
#    return (y - output) * output * (1 - output)

# even more simple, using the function above
#def error_term_formula(x, y, output):
#    return (y - output) * sigmoid_prime(output)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

# Training function
def train_nn(features, targets, epochs, learnrate):

    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1/n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w = np.add(del_w, error_term * x)

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
        weights = np.add(weights, learnrate * del_w / n_records)

        # Printing out the log-loss error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))

            # We are using binary cross-entropy (log-loss) to monitor training progress
            # as well as for computing gradients, to stay consistent with the loss function
            # used in training.
            loss = np.mean(error_formula(targets, out))
            print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return weights

weights = train_nn(features, targets, epochs, learnrate)


################################
## Calculate the Accuracy on the Test Data

# Calculate accuracy on test data
test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print(f"Prediction accuracy: {accuracy:.3f}")