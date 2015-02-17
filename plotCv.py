import os
import argparse
import numpy as np 
import matplotlib.pyplot as plt
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--name', type=str, required=True, help='Name of directory.')
    parser.add_argument('--temps', type=str, default="short", help='Optional. Title for plot.')
    args = parser.parse_args()

    name = args.name
    temps = args.temps

    os.chdir("%s_wham" % temps)

    Cv = np.loadtxt("cv",usecols=(0,1))
    maxindx = list(Cv[:,1]).index(max(Cv[:,1]))
    Tf = Cv[maxindx,0]

    QvsT = np.loadtxt("Q_vs_T",dtype=float)
    QvsT[:,1] = QvsT[:,1]/max(QvsT[:,1])
    dQdT = np.array([ abs((QvsT[i+1,1] - QvsT[i,1])/(QvsT[i+1,0] - QvsT[i,0])) for i in range(len(QvsT[:,0])-1) ])
    Tmidpoints = np.array([ 0.5*(QvsT[i+1,0] + QvsT[i,0]) for i in range(len(QvsT[:,0])-1) ])
    
    print "  Folding temperature:   %.2f K" % Tf
    try:
        maxdQdT = dQdT[maxindx - 1]
        rightsearch = maxindx - 1
        for i in range(len(dQdT)):
            if dQdT[rightsearch] <= (maxdQdT/2.):
                break
            else:
                rightsearch += 1
        leftsearch = maxindx - 1
        for i in range(len(dQdT)):
            if dQdT[leftsearch] <= (maxdQdT/2.):
                break
            else:
                leftsearch -= 1
        deltaT_FWHM = QvsT[rightsearch,0] - QvsT[leftsearch,0]
        Omega = ((Tf**2)/(deltaT_FWHM))*maxdQdT
            
        print "  Folding cooperativity: %.2f " % Omega
        error = 0
    except:
        print "  couldn't determine folding cooperativity"
        error = 1

    files = glob.glob("free*[0-9][0-9][0-9]")
    errF = [ abs(float(x.split("free")[1]) - Tf*10)  for x in files ]
    F = np.loadtxt(files[errF.index(min(errF))])
    #third = int(len(F)/5)
    #maxF = max([max(F[third - 20:,1]),max(F[:-third + 20,1])])
    plt.figure()
    plt.plot(F[:,0],F[:,1],'b',lw=2)
    plt.ylim(0,9)
    plt.title("Free energy %s" % name,fontsize=20)
    plt.xlabel("Q",fontsize=20)
    plt.ylabel("F(Q)",fontsize=20)
    plt.savefig("F_vs_Q.pdf")
    plt.savefig("F_vs_Q.png")


    plt.figure()
    plt.plot(Cv[:,0],Cv[:,1],'r',lw=2)
    plt.xlabel("Temperature (K)")
    plt.ylabel("Heat Capacity (kJ/mol K)")
    plt.title("$C_v(T)$ for %s" % name)
    plt.savefig("cv.pdf")
    plt.savefig("cv.png")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(QvsT[:,0],QvsT[:,1],'b',lw=2)
    ax1.set_ylim(0,1)
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("$\\left< Q \\right>(T)$")
    if error == 0:
        ax1.set_title("$\\left< Q \\right>(T)$   $\\Omega_c = %.2f$  for %s" % (Omega,name))
    else:
        ax1.set_title("$\\left< Q \\right>(T)$  for %s" % name)

    ax2.plot(QvsT[:-1,0],dQdT,'g',lw=2)
    ax2.set_ylabel("$\\left|\\frac{d\\left< Q \\right>}{dT}\\right|$",rotation="horizontal",fontsize=20)
    plt.savefig("QvsT.pdf")
    plt.savefig("QvsT.png")

    os.chdir("..")
    if temps == "long":
        print "  Wham done! Plotted Cv and melting curve: long_wham/cv_and_melt.pdf"
        open("long_Tf","w").write("%.2f" % Tf)
    else:
        print "  Wham done! Plotted Cv and melting curve: short_wham/cv_and_melt.pdf"
        open("short_Tf","w").write("%.2f" % Tf)
    plt.show()
