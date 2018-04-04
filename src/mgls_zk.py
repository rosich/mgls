import numpy as np
from EnvGlobals import Globals

def gls():
    #try to import astroML package
    try:
        import astroML.time_series
    except ImportError:
        print "astroML package is needed."

    print "Z-K implementation"
    #omega = [Globals.freq_range[0] + i*0.00001 for i in range(int((Globals.freq_range[1]-Globals.freq_range[0])/0.00001))  ]
    #omega = np.array(omega)
    #omega = 2.0*pi*omega
    steps = 50000
    period = np.linspace(Globals.period_range[0], Globals.period_range[1], steps)
    omega = 2 * np.pi / period
    PS = astroML.time_series.lomb_scargle(Globals.time, Globals.rv, Globals.rv_err, \
                                            omega, generalized=True)
    print np.argmax(PS)*(Globals.period_range[1] - Globals.period_range[0])/steps + \
          Globals.period_range[0], max(PS)
    # Get significance via bootstrap
    D = astroML.time_series.lomb_scargle_bootstrap(Globals.time, Globals.rv, Globals.rv_err, omega, \
                                                    generalized=True, N_bootstraps=10000, random_state=0)

   
    sig1, sig5 = np.percentile(D, [99.9, 99.])
    print sig1, sig5
