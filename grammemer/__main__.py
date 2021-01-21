import sys
from grammemer import Grammemer




if len(sys.argv) > 1:
    grammemer = Grammemer()
    grammemer.load()    
    x = sys.argv[1]
    if len(sys.argv) > 2:
        path = sys.argv[2]
    else:
        path = None    
    grammemer(x, path)
else:
    print('Text not found.')