# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 01:27:16 2015

@author: Vahndi


"""

class a(object):

    def __init__(self, flag):
        
        self._flag = flag
        
    def checkFlag(self):
        
        print self._flag
        


f = True

obj = a(f)

f = False
obj.checkFlag()



