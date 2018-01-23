import math
import numpy as np

def branin(params):    
    x = params[0]
    y = params[1]
    
    result = np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + 
         (5/math.pi)*x - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10
    
    result = float(result)
    
    #print('Result = %f' % result)
    #time.sleep(np.random.randint(60))
    return result