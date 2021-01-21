'''
Набор свойств для классов:

    - data.Data
'''
@property
def frame(self):
    return self._frame

@frame.setter
def frame(self, value):
    self._frame = value

@frame.deleter
def frame(self):
    del self._frame

@property
def x(self):
    return self._x

@x.setter
def x(self, value):
    self._x = value

@x.deleter
def x(self):
    del self._x

@property
def y(self):
    return self._y

@y.setter
def y(self, value):
    self._y = value

@y.deleter
def y(self):
    del self._y


'''
EOF
'''