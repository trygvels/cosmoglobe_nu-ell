import numpy as np

c = 299792458.0
kb =1.3806488e-23
h = 6.62606957e-34
Tcmb = 2.72548

def brightness(T, nu):
    """
    T in Kelvins
    nu in Hz

    returns brightness in Watts per square meter per Hertz per steradian
    """

    b_nu = (2 * h * nu**3 / c**2) / (np.exp(h * nu / (kb * T)) - 1)

    return b_nu


def dbdt(T, nu):
    """
    T in Kelvins
    nu in Hz

    returns brightness in Watts per square meter per Hertz per steradian
    """

    x = h * nu / (kb * T)
    ans = 2 * np.exp(x) * x**2 * nu**2 * kb / c**2 / (np.exp(x) - 1)**2

    return ans
    


def foo():
    import matplotlib.pyplot as mpl

    nu = np.arange(1, 500) * 1e9
    t = 2.72548

    b = brightness(t, nu)

    mpl.plot(nu, b)
    mpl.show()


def foo2():
    import matplotlib.pyplot as mpl
    nu = np.arange(1, 500) * 1e9
    t1 = 2.725
    t2 = 2.726

    deriv = dbdt(t1, nu)
    b1 = brightness(t1, nu)
    b2 = brightness(t2, nu)
    deriv2 = (b2 - b1) / (t2 - t1)

    mpl.plot(nu, deriv)
    mpl.plot(nu, deriv2)
    mpl.show()


def a2t(nu, t_cmb=Tcmb):
    """
    Antenna to thermodynamic temperature fluctuation conversion.
    Input is frequency in Hz.
    """

    x = h * nu / (kb * t_cmb)
    return (np.exp(x) - 1)**2 / (x**2 * np.exp(x))


def thermal_sz(nu):
    x = h * nu / (kb * Tcmb)
    return (x * (np.exp(x) + 1) / (np.exp(x) - 1) - 4)**2


def foo3():
    import matplotlib.pyplot as mpl
    nu1 = 100.e9
    nu2 = 143.e9
    nu3 = 217.e9


    nu = np.arange(1000) * 1.0e9

    mpl.plot(nu, thermal_sz(nu))
    mpl.show()


if __name__ == "__main__":
    #foo()
    #foo2()
    foo3()
