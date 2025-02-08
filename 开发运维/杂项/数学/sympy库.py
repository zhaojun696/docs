#%%

from sympy import symbols,Eq, solve
x = symbols('x')
equation = Eq(x**2 - 4, 0)
solution = solve(equation, x)
solution


# %%
import sympy as sp
from sympy import latex

# 定义符号变量
a = sp.Symbol('a')
b = sp.Symbol('b')
c = a + b**2

# 打印为LaTeX格式
latex_code = latex(c)
latex_code

#%%
import sympy as sp
from sympy import latex

# 定义符号变量
a = sp.Symbol('a')
b = sp.Symbol('b')



c = a + b**2

def my_function(x):
    return x + x**2

c=c+my_function(a)

# 打印为LaTeX格式
latex_code = latex(c)
latex_code

#%%
from sympy import symbols, summation

x = symbols('x')
expr = summation(x**2, (x, 0, 5))
latex_code = latex(expr)
latex_code

#%%
import sympy
from sympy.abc import *
f= (1/2)**sympy.log(n)
expr=sympy.summation(f,(n,1,sympy.oo)) #如下左图所示
expr
#%%
import sympy
n = symbols('n')
f= 1/sympy.log(n) +3
expr=summation(f,(n,1,sympy.oo))
latex_code = latex(expr)
latex_code
expr
#%%
import sympy
#后续在进行求导、积分等运算时，会用到大量函数表达的形式，sympy同时也提供了声明函数的方法
#sympy之所以相对于符号又单独提出函数的形式，主要是为了后面求解微积分及方程时，让代码更加直观
#或者
f=symbols('f',function=True)

x=symbols('x')
latex_code = latex(x+f)
latex_code

