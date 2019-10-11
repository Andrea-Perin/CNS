# -*- coding: utf-8 -*-

def my_function():
    print('This is just an example')
    return 'OK'
    
    
if __name__ == '__main__':
    
    print('This code is not executed if this file has been imported')
    print('This is very useful, for example to test the functions implemented in the module')
    
    # Prepare a test for my_function
    if my_function() == 'OK':
        print('TEST PASSED')
    else:
        print('ERROR! TEST FAILED!')
        