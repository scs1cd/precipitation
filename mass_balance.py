import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

AO, AMg, AFe, ASi         = 16, 24, 56, 28

def get_crit_loc(Tcmb, X_init, X):
    return (np.abs(X - X_init)).argmin()

def mole2mass_core(cbarFe, cbarO, cbarSi, cbarMg):
    """Convert mole fractions to mass fractions assuming 
    an Fe-O-Si-Mg core. """

    Abar = cbarO*AO + cbarSi*ASi + cbarMg*AMg + cbarFe*AFe
    cO   = cbarO  * AO  /Abar
    cSi  = cbarSi * ASi /Abar
    cMg  = cbarMg * AMg /Abar   
    return cO,cSi,cMg

def mole2mass_mant(cbarFeO, cbarSiO2, cbarMgO):
    """Convert mole fractions to mass fractions assuming 
    an FeO-SiO2-MgO mantle. """
    
    AMgO = AMg + AO
    AFeO = AFe + AO
    ASiO2= ASi + AO*2    
    Abar = cbarFeO*AFeO + cbarSiO2*ASiO2 + cbarMgO*AMgO
    cFeO = cbarFeO  * AFeO  /Abar
    cSiO2= cbarSiO2 * ASiO2 /Abar
    cMgO = cbarMgO  * AMgO  /Abar   
    return cFeO,cSiO2,cMgO

def mass2moleconc_core(cO, cSi, cMg):
    """Convert mass fractions to mole fractions assuming 
    an Fe-O-Si-Mg core. """
    
    Abar = 1 / (cO/AO + cSi/ASi + cMg/AMg + (1-cO-cSi-cMg)/AFe)
    cbarO  = cO  / (AO  /Abar)
    cbarSi = cSi / (ASi /Abar)
    cbarMg = cMg / (AMg /Abar)    
    return cbarO, cbarSi, cbarMg

def get_molefractions(M):
    """Compute mole fractions given the total number of 
    moles of all constituents."""

    MFe, MO, MSi, MMg = M[0],M[1],M[2],M[3]
    MFeO, MMgO, MSiO2 = M[4],M[5],M[6]
    
    XO    = MO /(MFe+MO+MSi+MMg)
    XFe   = MFe/(MFe+MO+MSi+MMg)
    XSi   = MSi/(MFe+MO+MSi+MMg)
    XMg   = MMg/(MFe+MO+MSi+MMg)
    XFeO  = MFeO /(MFeO+MMgO+MSiO2)
    XMgO  = MMgO /(MFeO+MMgO+MSiO2)
    XSiO2 = MSiO2/(MFeO+MMgO+MSiO2)

    return [XFe, XO, XSi, XMg, XFeO, XMgO, XSiO2]
    
def logKd(a, b, c, Tcmb): 
    """Distribution coefficient Kd. 
    Pressure is set to 135 GPa. """

    P = 135.0
    return a + (b/Tcmb) + (c*P/Tcmb)

def dissolution_closure():

    params = {}
        
    def dissolution(xp, KMg, KO, KSi, a, b, c, d, x, y, z):
    
        """Mass balance following for dissolution reactions following
        the Supplement in Rubie et al 2011.
        a = Fe
        b = Mg 
        c = O
        d = Si
        x = FeO
        y = MgO 
        z = SiO2
    
        Primes (p) denote new molar concentrations; unprimed variables are the current concentration. 
        """ 
        
        yp = xp*(y+b) / ( (x+a-xp)*(KMg/KO) + xp )
        
        ap    = x  + a - xp
        bp    = y  + b - yp
                
        print(y, yp, b, bp) 
           
        alpha = z  + d 
        gamma = ap + bp + x + y + 3*z + c + d - xp - yp
        sigma = x + y + 2*z + c - xp - yp
       
        A = ( 3.0*ap**2 + 2*xp*KO/KSi )
        B =-( 2*alpha*xp + xp*sigma )*KO/KSi + ap*gamma 
        C = xp*alpha*sigma * KO/KSi
        
        zpp = (-B + np.sqrt(B**2 - 4.0*A*C) ) / (2*A)
        zpm = (-B - np.sqrt(B**2 - 4.0*A*C) ) / (2*A)
            
        # Assume zpm is the correct root
        zp = zpm
        cp = x + y + 2*z + c - xp - yp - 2*zp
        dp = z + d - zp

        # Check that zpm balances total mass
        tot_in  = 2*x +2*y +3*z  +a +b +c +d 
        tot_zpp = 2*xp+2*yp+3*zpp+ap+bp+cp+dp
        tot_zpm = 2*xp+2*yp+3*zpm+ap+bp+ (x+y+2*z+c-xp-yp-2*zpm) + (z+d-zpm)
         
        if np.round(tot_in-tot_zpp,0) != 0:
            zp = zpm
            cp = x + y + 2*z + c - xp - yp - 2*zp
            dp = z + d - zp
                
        XO_c    = cp/(ap+bp+cp+dp)
        XFe_c   = ap/(ap+bp+cp+dp)
        XFeO_m  = xp/(xp+yp+zp)
               
        KO_new  = XO_c * XFe_c / XFeO_m
        KSi_new = (xp**2 * dp*(ap+bp+cp+dp))/(ap**2 * zp*(xp+yp+zp))
        KMg_new = xp*bp/(yp*ap)
    
        # Shows # moles are conserved for each species
        #print('Fe_tot = ', x+a, xp+ap, 'Si_tot = ', z+d, zp+dp) 
        #print('O_tot  = ', x+y+2*z+c , xp+yp+2*zp+cp , 'Mg_tot = ', y+b,yp+bp) 
        
        params['Fe']   = ap
        params['Mg']   = bp    
        params['O']    = cp
        params['Si']   = dp
        params['FeO']  = xp
        params['MgO']  = yp       
        params['SiO2'] = zp
        params['KO']   = KO
        params['KSi']  = KSi
        params['KO_new']  = KO_new
        params['KSi_new'] = KSi_new
        params['KMg_new'] = KMg_new
        
        #print this out if you want to chk that the -ve root is conserving mass. 
        #It is in all calculations I have looked at. 
        #print('Mass in,out_z+,out_z-,KO_old,KO_new = {:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(tot_in, tot_zpp, tot_zpm, KO, KO_new))

        return KO - KO_new
    
    return params, dissolution

def rubie_closure():

    params = {}
        
    def rubie(xp, KMg, KO, KSi, a, b, c, d, x, y, z):
    
        """Mass balance following the Supplement in Rubie et al 2011.
        a = Fe
        b = Mg 
        c = O
        d = Si
        x = FeO
        y = MgO 
        z = SiO2
    
        Primes (p) denote new molar concentrations; unprimed variables are the current concentration. 
        """ 
        
        yp = xp*(y+b) / ( (x+a-xp)*KMg + xp )
        
        ap    = x  + a - xp
        bp    = y  + b - yp
                
        alpha = z  + d 
        gamma = ap + bp + x + y + 3*z + c + d - xp - yp
        sigma = xp + yp
       
        A = ( 3.0*xp**2 - KSi*(ap**2) )
        B =-( KSi*(ap**2)*sigma + 3.0*(xp**2)*alpha + (xp)**2*gamma )
        C = (xp)**2*alpha*gamma
        
        zpp = (-B + np.sqrt(B**2 - 4.0*A*C) ) / (2*A)
        zpm = (-B - np.sqrt(B**2 - 4.0*A*C) ) / (2*A)
            
        # Assume zpm is the correct root
        zp = zpm
        cp = x + y + 2*z + c - xp - yp - 2*zp
        dp = z + d - zp

        # Check that zpm balances total mass
        tot_in  = 2*x +2*y +3*z  +a +b +c +d 
        tot_zpp = 2*xp+2*yp+3*zpp+ap+bp+cp+dp
        tot_zpm = 2*xp+2*yp+3*zpm+ap+bp+ (x+y+2*z+c-xp-yp-2*zpm) + (z+d-zpm)
         
        if np.round(tot_in-tot_zpp,0) != 0:
            zp = zpm
            cp = x + y + 2*z + c - xp - yp - 2*zp
            dp = z + d - zp
                
        XO_c    = cp/(ap+bp+cp+dp)
        XFe_c   = ap/(ap+bp+cp+dp)
        XFeO_m  = xp/(xp+yp+zp)
               
        KO_new  = XO_c * XFe_c / XFeO_m
        KSi_new = (xp**2 * dp*(ap+bp+cp+dp))/(ap**2 * zp*(xp+yp+zp))
        KMg_new = xp*bp/(yp*ap)
    
        # Shows # moles are conserved for each species
        #print('Fe_tot = ', x+a, xp+ap, 'Si_tot = ', z+d, zp+dp) 
        #print('O_tot  = ', x+y+2*z+c , xp+yp+2*zp+cp , 'Mg_tot = ', y+b,yp+bp) 
        
        params['Fe']   = ap
        params['Mg']   = bp    
        params['O']    = cp
        params['Si']   = dp
        params['FeO']  = xp
        params['MgO']  = yp       
        params['SiO2'] = zp
        params['KO']   = KO
        params['KSi']  = KSi
        params['KO_new']  = KO_new
        params['KSi_new'] = KSi_new
        params['KMg_new'] = KMg_new
        
        #print this out if you want to chk that the -ve root is conserving mass. 
        #It is in all calculations I have looked at. 
        #print('Mass in,out_z+,out_z-,KO_old,KO_new = {:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(tot_in, tot_zpp, tot_zpm, KO, KO_new))

        return KO - KO_new
    
    return params, rubie

def dissolution_closure():

    params = {}
               
    def dissolution(xp, KMg, KO, KSi, a, b, c, d, x, y, z):
    
        """Mass balance following for dissolution reactions following
        the Supplement in Rubie et al 2011.
        a = Fe
        b = Mg 
        c = O
        d = Si
        x = FeO
        y = MgO 
        z = SiO2
    
        Primes (p) denote new molar concentrations; unprimed variables are the current concentration. 
        """ 
        
        yp = xp*(y+b) / ( (x+a-xp)*(KMg/KO) + xp )
        
        ap    = x  + a - xp
        bp    = y  + b - yp
                
        alpha = z  + d 
        gamma = ap + bp + x + y + 3*z + c + d - xp - yp
        sigma = x + y + 2*z + c - xp - yp
       
        A = ( 3.0*ap + 2*xp*KO/KSi )
        B =-( 2*alpha*xp + xp*sigma )*KO/KSi - ap*gamma 
        C = xp*alpha*sigma * KO/KSi
        
        zpp = (-B + np.sqrt(B**2 - 4.0*A*C) ) / (2*A)
        zpm = (-B - np.sqrt(B**2 - 4.0*A*C) ) / (2*A)
            
        # Assume zpm is the correct root
        zp = zpm
        cp = x + y + 2*z + c - xp - yp - 2*zp
        dp = z + d - zp

        # Check that zpm balances total mass
        tot_in  = 2*x +2*y +3*z  +a +b +c +d 
        tot_zpp = 2*xp+2*yp+3*zpp+ap+bp+cp+dp
        tot_zpm = 2*xp+2*yp+3*zpm+ap+bp+ (x+y+2*z+c-xp-yp-2*zpm) + (z+d-zpm)
         
        if np.round(tot_in-tot_zpp,0) != 0:
            zp = zpm
            cp = x + y + 2*z + c - xp - yp - 2*zp
            dp = z + d - zp
                
        XFe_c   = ap/(ap+bp+cp+dp)
        XMg_c   = bp/(ap+bp+cp+dp)
        XO_c    = cp/(ap+bp+cp+dp)
        XSi_c   = dp/(ap+bp+cp+dp)
        XFeO_m  = xp/(xp+yp+zp)
        XMgO_m  = yp/(xp+yp+zp)
        XSiO2_m = zp/(xp+yp+zp)        
               
        KO_new  = XO_c * XFe_c / XFeO_m
        KSi_new = XSi_c * XO_c**2 / XSiO2_m
        KMg_new = XO_c * XMg_c / XMgO_m
    
        # Shows # moles are conserved for each species
        #print('Fe_tot = ', x+a, xp+ap, 'Si_tot = ', z+d, zp+dp) 
        #print('O_tot  = ', x+y+2*z+c , xp+yp+2*zp+cp , 'Mg_tot = ', y+b,yp+bp) 
        
        params['Fe']   = ap
        params['Mg']   = bp    
        params['O']    = cp
        params['Si']   = dp
        params['FeO']  = xp
        params['MgO']  = yp       
        params['SiO2'] = zp
        params['KO']   = KO
        params['KSi']  = KSi
        params['KO_new']  = KO_new
        params['KSi_new'] = KSi_new
        params['KMg_new'] = KMg_new
        
        #print this out if you want to chk that the -ve root is conserving mass. 
        #It is in all calculations I have looked at. 
        #print('Mass in,out_z+,out_z-,KO_old,KO_new = {:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}'.format(tot_in, tot_zpp, tot_zpm, KO, KO_new))

        return KO - KO_new
    
    return params, dissolution
    
    # Rubie assumes exchange for MgO and SiO2 and dissolution for FeO

# This example shows that the method works when the Kd are set to match the input compositions

def rubie_example():

    T = 5500

    Mtot = 100
    MFe  = 88#XFe_c * Mtot
    MO   = 5 #XO_c  * Mtot
    MSi  = 5 #XSi_c * Mtot
    MMg  = 2 #XMg_c * Mtot

    #25 FeO, 50 MgO, 25 SiO2 = 25+50+50=125 O
    #25 Fe , 50 Mg , 25 Si = 100
    MFeO = 0.25  * Mtot
    MMgO = 0.5   * Mtot
    MSiO2= 0.25  * Mtot

    XO_c    = MO /(MFe+MO+MSi+MMg)
    XFe_c   = MFe/(MFe+MO+MSi+MMg)
    XSi_c   = MSi/(MFe+MO+MSi+MMg)
    XMg_c   = MMg/(MFe+MO+MSi+MMg)
    XFeO_m  = MFeO /(MFeO+MMgO+MSiO2)
    XMgO_m  = MMgO /(MFeO+MMgO+MSiO2)
    XSiO2_m = MSiO2/(MFeO+MMgO+MSiO2)

    Kd_O    = XO_c  * XFe_c / XFeO_m
    Kd_Mg   = XMg_c * XFeO_m / (XMgO_m * XFe_c)
    Kd_Si   = XSi_c * XFeO_m**2 / (XSiO2_m * XFe_c**2)

    print(XFe_c, XO_c, XMg_c, XSi_c)
    print(XFeO_m, XMgO_m, XSiO2_m)
    print(Kd_O, Kd_Mg, Kd_Si)

    params, rubie = rubie_closure()
    rubie(MFeO, Kd_Mg, Kd_O, Kd_Si, MFe, MMg, MO, MSi, MFeO, MMgO, MSiO2)
    print('SiO2 = ', params['SiO2'], MSiO2)
    print('FeO  = ', params['FeO'] , MFeO) 
    print('MgO  = ', params['MgO'] , MMgO)  
    print('Fe   = ', params['Fe']  , MFe)
    print('Si   = ', params['Si']  , MSi)
    print('O    = ', params['O']   , MO)
    print('Mg   = ', params['Mg']  , MMg)
    print('Kd_Si= ', params['KSi_new']  , Kd_Si)
    print('Kd_Mg= ', params['KMg_new']  , Kd_Mg)
    print('Kd_O = ', params['KO_new'], Kd_O)
    
    return

def run_massbalance(M, K, Tcmb, var):
        
    """Set up the inputs to the function rubie:
    Kd for Mg, O and Si, Tcmb and Mole fractions of all species"""

    if var == 1:
        params, rubie = rubie_closure()
    else:
        print('Dissolution')
        params, rubie = dissolution_closure()
    
    MFe, MO, MSi, MMg = M[0]*np.ones(len(Tcmb)),M[1]*np.ones(len(Tcmb)),M[2]*np.ones(len(Tcmb)),M[3]*np.ones(len(Tcmb))
    MFeO, MMgO, MSiO2 = M[4]*np.ones(len(Tcmb)),M[5]*np.ones(len(Tcmb)),M[6]*np.ones(len(Tcmb))
    
    Kd_Mg, Kd_O, Kd_Si = K[0], K[1], K[2]
    
    # Initial concentrations
    MFei, MOi, MSii, MMgi = MFe[0] , MO[0]  , MSi[0]  , MMg[0]
    MFeOi, MMgOi, MSiO2i  = MFeO[0], MMgO[0], MSiO2[0]
    
    iX = get_molefractions(M) # Returns [XFe, XO, XSi, XMg, XFeO, XMgO, XSiO2]
    cO_c_init  , cSi_c_init  , cMg_c_init  = mole2mass_core(iX[0], iX[1], iX[2], iX[3]) 
    cFeO_m_init, cSiO2_m_init, cMgO_m_init = mole2mass_mant(iX[4], iX[5], iX[6]) 
    iC = [0.0, cO_c_init, cSi_c_init, cMg_c_init, cFeO_m_init, cMgO_m_init, cSiO2_m_init]
    
    tt = 0
    for t in Tcmb:         
            
        optimize.brentq(rubie, 0.0001, 100, args=(Kd_Mg[tt], Kd_O[tt], Kd_Si[tt], MFei, 
                                                  MMgi, MOi, MSii, MFeOi, MMgOi, MSiO2i))

        MFe[tt] , MO[tt]  , MSi[tt]  , MMg[tt] = params['Fe'] , params['O']  , params['Si']  , params['Mg']
        MFeO[tt], MMgO[tt], MSiO2[tt]          = params['FeO'], params['MgO'], params['SiO2']
            
        # Set new initial condition - but doesn't seem to matter... 
        MFei, MOi, MSii, MMgi = params['Fe'] , params['O']  , params['Si']  , params['Mg']
        MFeOi, MMgOi, MSiO2i  = params['FeO'], params['MgO'], params['SiO2']
        
        tt = tt + 1  

    # Output compositions
    oM = [MFe, MO, MSi, MMg, MFeO, MMgO, MSiO2]
    oX = get_molefractions(oM) 
    cO_c1  , cSi_c1  , cMg_c1  = mole2mass_core(oX[0], oX[1], oX[2], oX[3]) 
    cFeO_m1, cSiO2_m1, cMgO_m1 = mole2mass_mant(oX[4], oX[5], oX[6]) 
    oC = [0.0, cO_c1, cSi_c1, cMg_c1, cFeO_m1, cMgO_m1, cSiO2_m1]

    critloc = get_crit_loc(Tcmb, iX[3], oX[3])
     
    return iX, oX, iC, oC, critloc, oM

def get_crit_loc(Tcmb, X_init, X):
    return (np.abs(X - X_init)).argmin()

def print_init_conc(X,c):
    cFe, cO, cSi, cMg = c[0],c[1],c[2],c[3]
    cFeO, cMgO, cSiO2 = c[4],c[5],c[6]
    
    print('Initial core mole fractions of Fe  = {:6.2f} O   = {:6.2f} Mg   = {:6.2f} Si  = {:6.2f} All = {:6.2f}'
      .format(X[0],X[1],X[2],X[3],X[0]+X[1]+X[2]+X[3]))
    print('Initial mant mole fractions of FeO = {:6.2f} MgO = {:6.2f} SiO2 = {:6.2f} all = {:6.2f}'
      .format(X[4],X[5],X[6],X[4]+X[5]+X[6]))   
    print('Initial core mass fractions of O   = {:6.2f} Mg  = {:6.2f} Si  = {:6.2f}'
      .format(cO*100,cMg*100, cSi*100))
    print('Initial mant mass fractions of FeO = {:6.2f} MgO = {:6.2f} SiO2 = {:6.2f}'
      .format(cFeO*100, cMgO*100, cSiO2*100))
    
    return

def print_any_conc(X,c,loc):
    cFe, cO, cSi, cMg = c[0],c[1],c[2],c[3]
    cFeO, cMgO, cSiO2 = c[4],c[5],c[6]
    
    print('Core mole fractions of Fe  = {:6.2f} O   = {:6.2f} Mg   = {:6.2f} Si  = {:6.2f} All = {:6.2f}'
      .format(X[0][loc],X[1][loc],X[2][loc],X[3][loc],X[0][loc]+X[1][loc]+X[2][loc]+X[3][loc]))
    print('Mant mole fractions of FeO = {:6.2f} MgO = {:6.2f} SiO2 = {:6.2f} all = {:6.2f}'
      .format(X[4][loc],X[5][loc],X[6][loc],X[4][loc]+X[5][loc]+X[6][loc]))   
    print('Core mass fractions of O   = {:6.2f} Mg  = {:6.2f} Si  = {:6.2f}'
      .format(cO[loc]*100,cMg[loc]*100, cSi[loc]*100))
    print('Mant mass fractions of FeO = {:6.2f} MgO = {:6.2f} SiO2 = {:6.2f}'
      .format(cFeO[loc]*100, cMgO[loc]*100, cSiO2[loc]*100))
    
    return

# M = [iMFe, iMO, iMSi, iMMg, iMFeO, iMMgO, iMSiO2]
# get_molefractions(M) Returns [XFe, XO, XSi, XMg, XFeO, XMgO, XSiO2]
def plot_all(Tcmb, M, X, c, loc):

    dT = Tcmb[0]-Tcmb[1]
    
    MFe, MO, MSi, MMg = M[0],M[1],M[2],M[3]
    MFeO, MMgO, MSiO2 = M[4],M[5],M[6]

    XFe, XO, XSi, XMg = X[0],X[1],X[2],X[3]
    XFeO, XMgO, XSiO2 = X[4],X[5],X[6]

    cFe, cO, cSi, cMg = c[0],c[1],c[2],c[3]
    cFeO, cMgO, cSiO2 = c[4],c[5],c[6]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))

    ax1.set_xlim([6000,4000])
    ax1.set_xlabel("$T$ (K)")
    ax1.set_ylabel("Number of moles $M_{i}$ ")
    ax1.plot(Tcmb, MFe  , color='black' , label="Fe")
    ax1.plot(Tcmb, MMg  , color='red'   , label="Mg")
    ax1.plot(Tcmb, MO   , color='blue'  , label="O")
    ax1.plot(Tcmb, MSi  , color='orange', label="Si")
    ax1.plot(Tcmb, MMgO , color='red'   , label="MgO" , linestyle=':')
    ax1.plot(Tcmb, MFeO , color='blue'  , label="FeO" , linestyle=':')
    ax1.plot(Tcmb, MSiO2, color='orange', label="SiO2", linestyle=':')
    ax1.legend(loc=1)

    ax3.set_xlim([5000,4000])
    #ax3.set_ylim([0,3])       # To compare to the Du plot. 
    ax3.set_xlabel("$T$ (K)")
    ax3.set_ylabel("Weight percent $c_{i}$ (wt\%)")
    ax3.plot(Tcmb, cMg*100, color='red'   , label="Mg")
    ax3.plot(Tcmb, cO*100 , color='blue'  , label="O")
    ax3.plot(Tcmb, cSi*100, color='orange', label="Si")
    ax3.plot(Tcmb, cMgO*100 , color='red'   , label="MgO" , linestyle=':')
    ax3.plot(Tcmb, cFeO*100 , color='blue'  , label="FeO" , linestyle=':')
    #ax3.plot(Tcmb, cSiO2*100, color='orange', label="SiO2", linestyle=':')
    ax3.legend()

    ax2.set_xlim([5000,3000])
    ax2.set_xlabel("$T$ (K)")
    ax2.set_ylabel("Mole fractions $X_{i}$ ")
    ax2.plot(Tcmb, XFe  , color='black' , label="Fe")
    ax2.plot(Tcmb, XMg  , color='red'   , label="Mg")
    ax2.plot(Tcmb, XO   , color='blue'  , label="O")
    ax2.plot(Tcmb, XSi  , color='orange', label="Si")
    ax2.plot(Tcmb, XMgO , color='red'   , label="MgO" , linestyle=':')
    ax2.plot(Tcmb, XFeO , color='blue'  , label="FeO" , linestyle=':')
    ax2.plot(Tcmb, XSiO2, color='orange', label="SiO2", linestyle=':')
    ax2.plot(Tcmb[loc], XMg[loc]   , color='red'   , marker='o')
    ax2.legend(loc=2)

    dcdT_MgO = np.gradient(cMgO, dT)*1e5
    dcdT_FeO = np.gradient(cFeO, dT)*1e5
    
    ax4.set_xlim([6000,4000])
    ax4.set_xlabel("$T$ (K)")
    ax4.set_ylabel("Precipitation rate $dc_{i}/dT$ ($10^{-5}$/K)")
    ax4.plot(Tcmb, dcdT_MgO , color='red'   , label="MgO" , linestyle=':')
    ax4.plot(Tcmb, dcdT_FeO , color='blue'  , label="FeO" , linestyle=':')
    ax4.plot(Tcmb[loc], dcdT_MgO[loc]   , color='red'   , marker='o')
    ax4.legend(loc=2)