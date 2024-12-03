import sys
import numpy as np
#TODO, move read of parameters into this file

def read_g(fname ):
   """"
   read g tensor from the file
   """
   logf = open(fname).readlines()
   g = np.zeros([3,3])
   hfc = np.zeros([3,3])
   for i in range(3):
       temp = logf[i+1].split()
       g[i] = [float(x) for x in temp]

   indx = None
   for k, line in enumerate(logf):
       if 'hpc matrix' in line:
           idx = k
           break
   for i,line in enumerate(logf[k+1:k+4]):
       temp = line.split()
       hfc[i] = [float(x) for x in temp]

   zfs = np.zeros((3,3))

   return g, hfc, zfs

def get_modes(fname,fgeo,nmode=None):

    # load coordinates
    logdata = open(fgeo).readlines()
    nats = int(logdata[0])
    coords = []
    for i in range(2,2+nats):
        xyz = logdata[i].split()
        #print(xyz)
        coords.append(xyz)

    logdata = open(fname).readlines()
    logdata = logdata[1:]

    normal_modes = np.zeros((nmode,3*nats))
    for i in range(nmode):
        j = i*5 + 1
        line = logdata[j]
        mode_vec = line.split()
        mode_vec = np.array(mode_vec)
        mode_vec = mode_vec.astype(float)
        normal_modes[i,:] = mode_vec
    
    return coords, normal_modes


def write_normalmode_xsf(coords, normal_modes, printlist, Usys=None):
    from periodictable import elements
    nats = len(coords)
    
    if Usys is not None:
        nmode, nsys = Usys.shape

    for k in printlist:
        if Usys is None:
            output = open('xsf/normalmode_%d.xsf'%k,'w')
            vec = normal_modes[k]
        else:
            output = open('xsf/proj_normalmode_%d.xsf'%k,'w')
            vec = np.zeros(3*nats)
            for i in range(nmode):
                vec += normal_modes[i] * Usys[i,k].real
        output.write("ATOMS\n")

        j = 0 
        for i in range(nats):
            symbol = coords[i][0]
            artnumber = elements.symbol(symbol).number
            vx = vec[j]
            vy = vec[j+1]
            vz = vec[j+2]
            j += 3
            output.write("{}  {}  {}  {} {}  {}  {}\n".format(artnumber, coords[i][1],coords[i][2],coords[i][3], vx, vy, vz))
        output.close()


def get_phonon(fname, f2=None, f3=None, op=None):
   """
   read spin-phonon coupling elements for each mode
   """
   logvib = open(fname).readlines()
   if f2 is not None:
       log_hpc = open(f2).readlines()
   if f3 is not None:
       log_dip = open(f3).readlines()

   nvib = int((len(logvib)-1)/ 4)
   if op == 1: nvib = int((len(logvib)-1)/ 5)
   print('number of vibs=', nvib)

   freq = np.zeros(nvib)
   dgx = np.zeros((nvib, 3, 3))  # zeeman term
   if op == 1: palpha = None
   for i in range(nvib):
       k = i * 4 + 1
       if op == 1: k = i * 5 + 1
       line = logvib[k]
       freq[i] = float(line.split()[3])
       #print(i,freq[i])

       if op == 1:
           line = logvib[k+1]
           data = line.split()
           ndof = len(data)
           if palpha is None: 
               print('ndof=', ndof)
               palpha = np.zeros((ndof, nvib))
           for l in range(ndof):
               palpha[l,i] = float(data[l])

       # dgx
       shift = 0
       if op == 1: shift = 1
       dq = np.zeros((3,3))
       for l in range(3):
          line = logvib[k+l+1+shift]
          dq[l] = [float(g) for g in line.split()]
          #print(dq[l])
       dgx[i,:,:] = dq
       #dgx.append(dq)

   if op == 1:
       return freq, dgx, palpha
   return freq, dgx #, dAx, dDx

#def ohmic_spectrum(w):
#    if w == 0.0: # dephasing inducing noise
#        return gamma1
#    else: # relaxation inducing noise
#        return gamma1 / 2 * (w / (2 * np.pi)) * (w > 0.0)

#------------

