import numpy as np





def plot_CACA_CBCB_vs_Q():
     
    temps = [ x.rstrip("\n") for x in open("long_temps_last", "r").readlines() ]

        for i in range(len(temps)):
            T = temps[i]
            Q_temp = np.loadtxt("%s/Q.dat" % T)
            Qi_temp = np.loadtxt("%s/qimap.dat" % T)
            if i == 0:
                Qi = Qi_temp
                Q = Q_temp
            else:
                Qi = np.concatenate((Qi,Qi_temp),axis=0)
                Q = np.concatenate((Q,Q_temp),axis=0)




if __name__ == "__main__":

    plot_CACA_CBCB_vs_Q() 
