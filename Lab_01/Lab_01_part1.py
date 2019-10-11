# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Dr. Matteo Gadaleta

Lab. 01 - Part 1 - Python basics
 
"""

## ONLY IN SPYDER IDE: press F9 to execute the current line (or current selection)
## ONLY IN SPYDER IDE: press ctrl+Enter to execute the current cell (delimited by #%% and highligthed in the editor)

#%% The Zen of Python

import this


#%% Variables 

# No need to define the type of a variable
a = 1
b = 'string'

# A variable can change type at any time (dynamic type)
print(type(a))
a = 'string' # now a is a string
print(type(a))

# Comparison
print(a == b)
print(1 != 2)
print(1 > 2)
print(1 <= 2)

# Combine multiple comparisons
print(a == b and a == 'string')
# Personal advice: always use brackets to combine multiple comparisons, complex concatenations may generate unexpected results
print((a == b) and (type(a) is str) and (len(a) == 6) and ((a == 'string') or (type(a) is str)))

# Substring in string
print('ri' in a)
print('ro' in a)


#%% Built-in data structures

### LISTS
# Define a list
mylist = ['this', 'is', 'a', 'list']

# Access list elements (0 is the FIRST element)
print(mylist[0])
print(mylist[1])
print(mylist[4]) # <- ERROR - Out of range

# Negative index (VERY USEFUL!)
print(mylist[-1])
print(mylist[-2])

# Number of elements in a list
print(len(mylist))

# Append elements to a list
mylist.append('!')
# Now the list has 5 elements
print(mylist)
print(mylist[4])

# Insert elements to a specific index
mylist.insert(2, 'not')
print(mylist)

# Edit elements
mylist[2] = 'still'
print(mylist)

# It supports heterogeneous types
mylist.append(32.4)
print(mylist)
# You can also append the entire numpy library to a list!
import numpy as np
mylist.append(np)
print(mylist)
mylist[-1].array([1,2,3])

# Copy a list (PAY ATTENTION!)
mylist = ['this', 'is', 'a', 'list']
mylist2 = mylist
mylist[0] = 'Hey!'
print(mylist)
print(mylist2)
# mylist2 HAS BEEN ALSO MODIFIED!! Always check if the = operator assigns values or pointers!
# To make a different copy of a list use the .copy() function
mylist = ['this', 'is', 'a', 'list']
mylist2 = mylist.copy() # <- HERE
mylist[0] = 'Hey!'
print(mylist)
print(mylist2)


### TUPLES
# Tuples can be thought as immutable lists, they can't be changed in-place!
mytuple = ('this', 'is', 'a', 'tuple')
print(mytuple)
# NO item assignment
mytuple[0] = 'Hey!' # <- ERROR
# NO append
mytuple.append('!') # <- ERROR
# They must be redefined
mytuple = ('this', 'is', 'a', 'tuple', '!') # <- OK
print(mytuple)



#%% Conditional statements

# Check if a number is even or odd
a = 10
if a % 2 == 0:
    print('a is even')
else:
    print('a is odd')

# Select a color
a = 'blue'
if a == 'blue':
    print('Color blue selected')
elif a == 'red':
    print('Color red selected')
elif a == 'yelow':
    print('Color yellow selected')
else:
    print('Color not supported')

# not statement
a = 1
if not a == 2:
    print('a is NOT 2')
    
# Check if element in list
mylist = [1, 2, 3, 4]
if 4 in mylist:
    print('There is a 4 in mylist!')
else:
    print('NO 4 in mylist!')



#%% Loops

### For loops
for i in range(10): 
    # the function range create a generator from 0 to N, the for loops iterate each of this values
    print(i)

# For loops can also iterate list values (or tuples)
mylist = ['Still', 'the', 'same', 'list']
for list_element in mylist:
    print(list_element)
 
# Iterate through multiple lists at the same time
# Example: element-wise product
list_a = [1, 2, 3, 4]
list_b = [2, 3, 5, 6]
result = [] # Empty list
for a, b in zip(list_a, list_b):
    # a contains the element in list_a
    # b contains the element in list_b
    result.append(a * b)
print(result)
   
# Enumerate
mylist = ['Still', 'the', 'same', 'list']
for idx, list_element in enumerate(mylist):
    # Now idx contains the index of the list_element
    print('mylist contains the element "%s" at index %d' % (list_element, idx))


### While loops
a = 0
while True:
    if a > 10:
        print('STOP!')
        break # ends the loop immediately
    print('a = %d - Keep going' % a) # String formatting (C-style) the value of a goes to %d
    a += 1 # Short version of a = a + 1
    
    
#%% Dictionaries
    
### Key-values pairs
mydict = {
        'Name': 'Federico',
        'Surname': 'Chiariotti',
        'Age': '27'
        }

### Check dictionary keys
print(mydict.keys())
if 'Name' in mydict.keys():
    print('"Name" is a key of mydict')
    
### Check dictionary values
print(mydict.values())
if 'Federico' in mydict.values():
    print('"Federico" is a value of mydict')

### Check dictionary items
print(mydict.items())
# Iterate through items
for key, value in mydict.items():
    print('Key: %s -> Value: %s' % (key, value))

### Add additional key-value pairs
mydict['Sex'] = 'M'
print(mydict)

# The values of a dictionary can be of any type!
# The keys of a dictionary must be hashable! (for example a list cannot be a key)
advisor = {
        'Name': 'Andrea',
        'Surname': 'Zanella',
        'Position': 'Full professor'
        }
mydict = {
        'Name': 'Federico',
        'Surname': 'Chiariotti',
        'Age': '27'
        'Sex': 'M',
        'Advisor': advisor
        }


#%% Functions

### Simple function
def print_hello():
    print('Hello!')
    
print_hello()

### Mandatory input arguments
def print_message(message):
    print(message)
    
print_message('Please print this message')
print_message() # <- ERROR

### Optional input arguments
def print_message(message='Default message'):
    print(message)
    
print_message('Please print this message')
print_message() # <- Print the default value

### Return something
def mysum(a, b):
    return a + b

c = mysum(2, 3)
print(c)


### Functions have access to the global namespace, BUT BE CAREFUL!
def print_a_squared():
    print(a**2)
    
a = 2
print_a_squared()

### You can use "a" but you cannot modify it...
def change_a():
    a = 4 # This is another "a" (local variable), it only exists within this function
    
a = 2
print(a)
change_a()
print(a) # "a" has NOT been changed

### ...unless you define it as global variable in the local space
def change_a():
    global a
    a = 4 # This is another "a" (local variable), it only exists in this function
    
a = 2
print(a)
change_a()
print(a) # "a" has been changed

# PERSONAL ADVICE: try to avoid using the global namespace inside functions, unless strictly necessary (almost never!). This is a bad practice.

#%% Files

# Write to a file
myfile = open('filetest.txt', 'w') # Write mode
myfile.write('First line\n')
myfile.close()

# Remember to always close files, or better use the "with" statement
with open('filetest.txt', 'w') as myfile:
    myfile.write('First line\n')
# Same as before, but the file is automatically closed.
    
# Append to a file
with open('filetest.txt', 'a') as myfile: # Append mode
    myfile.write("Let's add another line\n")

# Read from a file
with open('filetest.txt', 'r') as myfile: # Read mode
    file_content = myfile.read() # Read the entire file
print(file_content)

with open('filetest.txt', 'r') as myfile: # Read mode
    file_content_list = myfile.readlines() # Split lines into a list
print(file_content_list)


#%% Modules

### Import the entire module
import numpy
a = numpy.array([1,2,3])

### Rename the imported module
import numpy as np
a = np.array([1,2,3])

### Only import a single submodule
from numpy import array
a = array([1,2,3])
# or...
from numpy import array as apple
a = apple([1,2,3])


### Import custom modules
from my_module import hello
hello.my_function()

# or just a single function
from my_module.hello import my_function
my_function()


#%% Classes

class MyClass:
    
    # Initialize method (This is a special function, and it is called every time a new object is created)
    def __init__(self, init_param):
        # Store the input parameter
        self.init_param = init_param
        # Without this statement the init_param would not be accessible after the execution of the __init__ function, since it is a local variable
        
    def print_param(self): # <- Every function in a class has the keyword "self" as first argument
        # "self" is used to access the variables and methods stored in the object itself
        print(self.init_param)

    def set_param(self, new_param):
        self.init_param = new_param
        self.print_param()


# Define a new object
myobject = MyClass(init_param=5) # The __init__ is executed
# init_param is stored in the object thanks to the initialize function
print(myobject.init_param)
# Access the methods
myobject.print_param()
myobject.set_param(7)


### Inheritance
class MyClassChild(MyClass): # <- Parent class
    
    def __init__(self, init_param):
        super().__init__(init_param) # super() call the parent class
    
    def double_param(self):
        self.set_param(self.init_param * 2)

# Define a new object
myobject_child = MyClassChild(init_param=5) # The __init__ is executed
# All the methods have been inherited
myobject_child.print_param()
myobject_child.set_param(7)
myobject_child.double_param()


### Callable Class
class AddFixedValue:
    
    def __init__(self, add_value):
        # Store a constant value in the object, defined when a new object is created
        self.add_value = add_value
        
    # "magic" function to make the object callable
    def __call__(self, input_value):
        # Add the value stored in the object to the input value
        return input_value + self.add_value
    
# Create a new callable object
add_fixed_value = AddFixedValue(add_value=3)
# Now we can directly call the object, and it adds to the input the predefined value (3)
result = add_fixed_value(10)
print(result)
result = add_fixed_value(15)
print(result)


#%% Numpy

import matplotlib.pyplot as plt
import numpy as np

### Define arrays
a = np.array([1,2,3])            # Create a rank 1 array
b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array

# Shapes
print('"a" shape:', a.shape)
print('"b" shape:', b.shape)


### Acecssing elemets
print(a[1])
# Negative indexing (like lists)
print(a[-1])
# Multiple indexes
print(b[0, 1])


### Most of the basic MATLAB functions are similar
print(np.zeros([3,3]))
print(np.ones([4,3]))
print(np.eye(5))


### Random module
num_samples = 10000
# Uniform
a = np.random.random(num_samples)
plt.figure()
plt.hist(a, 20)
plt.xlabel('Value')
plt.ylabel('Counts')
# Normal
a = np.random.randn(num_samples)
plt.figure()
plt.hist(a, 20)
plt.xlabel('Value')
plt.ylabel('Counts')
# Exponential
a = np.random.exponential(scale=1, size=num_samples)
plt.figure()
plt.hist(a, 20)
plt.xlabel('Value')
plt.ylabel('Counts')


### Indexing
a = np.array([0,1,2,3,4,5,6,7,8,9])
# From index 4 (included) to index 7 (excluded)
print(a[4:7])
# From index 5 to end
print(a[5:])
# First 5 values
print(a[:5])
# Last 5 values
print(a[-5:])


### PAY ATTENTION WHEN COPYING ARRAYS!!
a = np.array([1,2,3,4,5,6])
print('a:', a)
b = a
b[0] = 1999
print('a:', a)
# a HAS BEEN ALSO MODIFIED!! Always check if the = operator assigns values or pointers!
# To make a safe copy
b = a.copy()
# Now the two variables have been allocated in two different areas of the memory


### Data type
# Automatically infer the type
a = np.array([1,2,3,4,5,6])
print(a.dtype)
a = np.array([1.0,2,3,4,5,6])
print(a.dtype)
# Explicit declaration
a = np.array([1,2,3,4,5,6], dtype=np.float32)
print(a.dtype)


### Masking
a = np.array([1,2,3,4,5,6])
# Create mask
mask = a > 3
print(mask)
# Apply mask
a[mask] = 3
print(a)


### Matrix math
a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])
print('a = \n', a, '\nb = \n', b)
# Elementwise operations
c = a + b
print('Elementwise sum\n', c)
c = a - b
print('Elementwise difference\n', c)
c = a * b
print('Elementwise product\n', c)
c = a / b
print('Elementwise division\n', c)
# Matrix product
c = np.dot(a, b)
# or 
c = a.dot(b)
print('Matrix product\n', c)
# Transpose
c = a.T
print('Transpose\n', c)

