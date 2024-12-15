"""
Created on Mon Apr 12 11:22:27 2021

@author: Tony
"""


def getcontentstring(s):
    newc = " "
    left = s.rfind("-")
    if left > 0:
        newc = s[left+2:]
    return newc



def dobranches(rstring):
    branches = []
    branch = []
    for x in rstring.split('\n'):
        indent = x.count("|")
        #print(str(indent))
        content = getcontentstring(x)
        oldindent = len(branch) 
        if oldindent > indent:
            branch = branch[:indent-1]
        if "weight" not in content:
            branch.append(content)
        if "weight" in content:
            content = content[9:]
            branch.append(content)
            branches.append(branch)
    return branches