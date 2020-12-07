import numpy as np
import numpy.random as rd

def produto_interno(u, v):
	prod = 0

	for i in range(u.size):
		prod += u[i]*v[i]
	return prod

u = np.random.rand(10000)
v = np.random.rand(10000)

#%timeit produto_interno(u, v)
#%timeit np.dot(u, v)

v = np.array([1, 2, 3, 4])
print(v)
print(v.dtype)

v2 = np.array([1, 2, 3, 4], dtype='float64')

print(v2)
print(v2.dtype)

# shape
print(v.shape)
print(v2.shape)

v3 = np.array([1, 2, 3, 4])
v3.shape = (2,2)
print(v3)

v4 = np.array([1, 2, 3, 4]).reshape(2,2)
print(v4)

v5 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
v5 = v5.reshape(2, -1)
print(v5)

print('*' * 20)

v = np.array(range(50)).reshape(2, 5, 5)

print('Shape = ', v.shape)
print('Número de dimensões = ', v.ndim)
print('Número de elementos = ', v.size)
print('\nTensor v = \n', v)

print('*' * 20)

v = np.zeros((3, 3))
print('V = \n', v)
u = np.ones((3, 3))
print('u = \n', u)
d = np.diag([10, 10, 10])
print('d = \n', d)

print('*' * 20)

v = np.arange(0, 5, 0.5)
u = np.linspace(0, 5, 10)
print('v =', v)
print('u =', u)

print('*' * 20)

v = np.array(range(50)).reshape(2, 5, 5)
print(v)
print(v.flatten())

print('*' * 20)

v_iter = v.flat
for i in v_iter:
	print(i, end=' ')

print('*' * 20)

# Indexação de tensores

print(v[1])
print(v[0])


print('*' * 20)

v = np.array([10, 20, 30, 40]).reshape(2, 2)
print(v[1,1])


print('*' * 20)

v = np.arange(8).reshape(2, 2, 2)
print('v = \n', v)
print('\n v[0,0,1] = ', v[1,1,1])

print('*' * 20)

v = np.arange(10)
u = v[1:3]
print(u)

print('*' * 20)

a = np.arange(15).reshape(3, 5)
print(a)

subA = a[0:2, 2:4]
print(subA)

print('*' * 20)

u = np.array([2.0, 3.5, 4.0, -10.1])
v = u[[2, 3]]
print(v)

print('*' * 20)

a = np.arange(10).reshape(2, 5)
b = a[:, [2, 4]]
print('a = \n', a)
print('b = \n', b)

print('*' * 20)

v = np.array([10, 20, 30])
u = np.array([2, 2, 2])
w = u+v
print(w)
w = u*v
print(w)

print('*' * 20)

x = np.array([10, 20])
y = x**2
print(y)

print('*' * 20)

x = np.arange(10)
media = x.mean()
menor_valor = np.min(x)
arg_max = np.argmax(x)
print('média = ', media)
print('menor valor = ', menor_valor)
print('Arg max =', arg_max)

print('*' * 20)

a = np.array([10, 30, 40, 20]).reshape(2, 2)
menor = a.min()
menor_colunas = a.min(axis=0)
print('a = \n', a)
print('menor valor = ', menor)
print('menor valor em cada coluna', menor_colunas)

print('*' * 20)

w = np.dot(u, v)
print('w = ', w)

x = u.dot(v)
print('x = ', x)

print('w = ', w)
x = 10 * w
print('x = ', x)

print('-' * 20)

a = np.array([[10, 20], [30, 40]])
print(a)

# criar uma matriz identidade

i = np.eye(5)
print(i)

# matriz diagonais

d = np.diag(np.arange(5))
print(d)

print('-' * 20)
## Operação sobre matrizes
v = np.array([10, 10])
a = np.arange(4).reshape(2, 2)
u = a.dot(v)
print(u)

print('-' * 20)

# multiplicação de matrizes

a = np.ones((2, 2))
print(a)
b = 10* np.ones((2, 2))
c = np.dot(a, b)
print(c)
# or
c = a @ b
print(c)

print('-' * 20)

a = np.arange(4).reshape(2, 2)
print(a)
print('transposta de a = \n', a.transpose())
print('transposta de a =\n', a.T)

print('-' * 20)

# funções universais

u = np.arange(5)
v = np.exp(u)
print(v)

print('-' * 20)

v = np.sin(u)
print(v)

print('-' * 20)

# operadores lógicos

u = np.arange(4).reshape(2, 2)
v = 2*np.ones((2,2))
w = u > v
print(w)

print('-' * 20)

u = np.array([-1, 2, -3])
v = np.array([True, False, True])
print(u[v])

w = u[u < 0]
print(w)

print('-' * 20)

print('u = ', u)
u[u<0] = 0
print('u = ', u)

print('-' * 20)

v = np.arange(5)
u = v >= 0
print('u =', u)
print(np.all(u))

print('-' * 20)

v = np.array([-1, 1, 2, 3])
u = v > 0
print('u =', u)
print(np.any(u))

print('-' * 20)

v = rd.rand(4, 4)
print(v)

print('-' * 20)

v = rd.normal(10, 1, (4, 4))
print(v)

print('-' * 20)

rd.seed(1000)
v = rd.rand(4)
print(v)
rd.seed(1000)
v = rd.rand(4)
print(v)

print('-' * 20)

a = np.array([10, 20, 30, 40]).reshape(2, 2)
b = np.array([5, 10])
x = np.linalg.solve(a, b)
print(x)

print('-' * 20)

b1 = np.array([5, 10]).reshape(2, 1)
b2 = np.array([5, 12]).reshape(2, 1)
b = np.hstack([b1, b2])

def fun(a, b1, b2):
	x1 = np.linalg.solve(a, b1)
	x2 = np.linalg.solve(a, b2)

a = np.random.rand(1000, 1000)
b1 = np.random.rand(1000).reshape(1000, 1)
b2 = np.random.rand(1000).reshape(1000, 1)
b = np.hstack([b1, b2])

print('-' * 20)

# matriz inversa
a = np.array([10, 20, 30, 40]).reshape(2, 2)
inv_a = np.linalg.inv(a)
print(a)
print(inv_a)

print('-' * 20)

rank_a = np.linalg.matrix_rank(a)
print('rank de a = ', rank_a)
det_a = np.linalg.det(a)
print(f'determinante de a = {det_a:.2f}')

print('-' * 20)

u = np.ones((4, 4))
v = np.array([10, 20, 30, 40]).reshape(1, 4)
x = u * v
print('u =\n', u)
print('v = \n', v)
print('x = \n', x)