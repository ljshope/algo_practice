import random

def onegame():
    flip = random.randint(1,6)
    if flip == 6:
        result = -4
    else:
        result = 1
    return result    
  
def invest( prob ):
    money = 100
    for i in range( 100 ):
        current = money
        if current >0: 
            money += onegame( )*int( prob*current )   
    return money

def checking():
    result = [0]*21
    for i in range( 1000 ):
        seq = range(21)
        for prob in seq:
            result[prob] += invest( prob/20 )/1000
    result = [ int(x) for x in result ]        
    print( result ) 
    
