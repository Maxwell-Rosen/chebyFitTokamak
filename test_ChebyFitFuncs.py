import chebyFitFuncs as chb
import numpy as np

func1 = lambda x: x**3
func1_der = lambda x: 3*x**2
func1_int = lambda x, a: 1/4*x**4 - 1/4*a**4

func2 = lambda x: x + x**2 + x**3 + x**4 + x**5 + x**6
func2_der = lambda x: 1 + 2*x + 3*x**2 + 4*x**3 + 5*x**4 + 6*x**5
func2_int = lambda x, a: \
    1/2*x**2 + 1/3*x**3 + 1/4*x**4 + 1/5*x**5 + 1/6*x**6 + 1/7*x**7 \
- (1/2*a**2 + 1/3*a**3 + 1/4*a**4 + 1/5*a**5 + 1/6*a**6 + 1/7*a**7)

func3 = lambda x: 2*x + 1
func3_der = lambda x: 2
func3_int = lambda x, a: x**2 + x - a**2/2 - a/2

func4 = lambda x, y: x**2 + y**2
func4_der_x = lambda x, y: 2*x
func4_der_y = lambda x, y: 2*y
func4_der_xy = lambda x, y: 0

func5 = lambda x, y: x**2 + y**2 + x*y
func5_der_x = lambda x, y: 2*x + y
func5_der_y = lambda x, y: 2*y + x
func5_der_xy = lambda x, y: 1   

func6 = lambda x, y: x**2 + y**2 + x*y + x**3 + y**3 + x**2*y + x*y**2
func6_der_x = lambda x, y: 2*x + y + 3*x**2 + 2*x*y + y**2
func6_der_y = lambda x, y: 2*y + x + 3*y**2 + 2*x*y + x**2  
func6_der_xy = lambda x, y: 1 + 2*y + 2*x

def test1():

    a, b, n = -1, 1, 5
    coeff = chb.coeff(a, b, n, func1)
    x = np.linspace(a,b,10)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff, x[i]), func1(x[i]))

    coeff_der = chb.coeffDer(a, b, n, coeff)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff_der, x[i]), func1_der(x[i]))

    coeff_int = chb.coeffInt(a, b, n, coeff)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff_int, x[i]), func1_int(x[i], a))

def test2():
    a, b, n = -5, 1, 10
    coeff = chb.coeff(a, b, n, func2)
    x = np.linspace(a,b,10)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff, x[i]), func2(x[i]))

    coeff_der = chb.coeffDer(a, b, n, coeff)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff_der, x[i]), func2_der(x[i]))

    coeff_int = chb.coeffInt(a, b, n, coeff)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff_int, x[i]), func2_int(x[i], a))

def test3():
    a, b, n = 0, 8, 3
    coeff = chb.coeff(a, b, n, func3)
    x = np.linspace(a,b,10)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff, x[i]), func3(x[i]))

    coeff_der = chb.coeffDer(a, b, n, coeff)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff_der, x[i]), func3_der(x[i]))

    coeff_int = chb.coeffInt(a, b, n, coeff)
    for i in range(len(x)):
        assert np.isclose(chb.fit(a, b, n, coeff_int, x[i]), func3_int(x[i], a))

def test4():
    a, b, c, d, n = -1, 1, -1, 1, 10
    coeff = chb.coeff2D(a, b, c, d, n, func4)
    x = np.linspace(a,b,10)
    y = np.linspace(c,d,10)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff, x[i], y[j]), func4(x[i], y[j]))

    coeff_der_x = chb.coeffPartialX(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_x, x[i], y[j]), func4_der_x(x[i], y[j]))

    coeff_der_y = chb.coeffPartialY(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_y, x[i], y[j]), func4_der_y(x[i], y[j]))

    coeff_der_xy = chb.coeffPartialXY(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_xy, x[i], y[j]), func4_der_xy(x[i], y[j]))

def test5():
    a, b, c, d, n = -1, 1, -1, 1, 10
    coeff = chb.coeff2D(a, b, c, d, n, func5)
    x = np.linspace(a,b,10)
    y = np.linspace(c,d,10)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff, x[i], y[j]), func5(x[i], y[j]))

    coeff_der_x = chb.coeffPartialX(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_x, x[i], y[j]), func5_der_x(x[i], y[j]))

    coeff_der_y = chb.coeffPartialY(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_y, x[i], y[j]), func5_der_y(x[i], y[j]))

    coeff_der_xy = chb.coeffPartialXY(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_xy, x[i], y[j]), func5_der_xy(x[i], y[j]))

def test6():
    a, b, c, d, n = -1, 1, -1, 1, 10
    coeff = chb.coeff2D(a, b, c, d, n, func6)
    x = np.linspace(a,b,10)
    y = np.linspace(c,d,10)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff, x[i], y[j]), func6(x[i], y[j]))

    coeff_der_x = chb.coeffPartialX(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_x, x[i], y[j]), func6_der_x(x[i], y[j]))

    coeff_der_y = chb.coeffPartialY(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_y, x[i], y[j]), func6_der_y(x[i], y[j]))

    coeff_der_xy = chb.coeffPartialXY(a, b, c, d, n, coeff)
    for i in range(len(x)):
        for j in range(len(y)):
            assert np.isclose(chb.fit2D(a, b, c, d, n, coeff_der_xy, x[i], y[j]), func6_der_xy(x[i], y[j]))


test1()
test2()
test3()
test4()
test5()
test6()
print("Passes all unit tests")