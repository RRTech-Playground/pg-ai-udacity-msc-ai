
# get and process input for a list of names
names =  input("Enter names separated by commas: ").title().split(",")

# get and process input for a list of the number of assignments
assignments = input("Enter assignment counts separated by commas: ").split(",")

# get and process input for a list of grades
grades = input("Enter grades separated by commas: ").split(",")

## message string to be used for each student
## HINT: use .format() with this string in your for loop
message = "Hi {},\n\nThis is a reminder that you have {} assignments left to \
submit before you can graduate. Your current grade is {} and can increase \
to {} if you submit all assignments before the due date.\n\n"

for name, assignment, grade in zip(names, assignments, grades):
    print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))