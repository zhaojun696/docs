#%%
!python -m pip install latexify-py -i https://mirror.baidu.com/pypi/simple

#%%
import latexify

@latexify.function
def ma10(a,b):
    c=a+b
    return c

ma10
#%%

def ma10(a,b):
    c=a+b
    return c

latexify.get_latex(ma10)

#%%
import math
import numpy as numpy
import latexify

# @latexify.get_latex
@latexify.expression
def f(x):
    if (x > 0):
        return x
    else:
        return 0

f
#%%
import math
import latexify

def ma(arr,peroid):
    pass

# @latexify.get_latex
@latexify.function
def 均线(x):
    x5=ma(x,5)
    x10=ma(x,10)
    return x5+x10
均线


#%%
from sympy import symbols, latex
 
# 定义符号变量 x、y
x = symbols('x')
y = symbols('y')
 
# 创建一个 SymPy 表达式
expr = (2*x + y)**3 - 5*x**4 + 10*y**2
 
# 将表达式转换为 LaTeX 字符串
latex_str = latex(expr)
 
latex_str
