import numpy as np

#Mole2mass and vice versa
AO, AMg, AFe, ASi = 16, 24, 56, 28

def mole2massconc_core(cbarO, cbarSi, cbarMgO):

    AMgO = AMg #+ AO
    Abar = cbarO*AO + cbarSi*ASi + cbarMgO*AMgO + (1-cbarO-cbarSi-cbarMgO)*AFe
    cO   = cbarO  * AO  /Abar
    cSi  = cbarSi * ASi /Abar
    cMgO = cbarMgO* AMgO/Abar   
    return cO,cSi,cMgO

def mass2moleconc_core(cO, cSi, cMgO):

    AMgO = AMg # AO
    Abar = 1 / (cO/AO + cSi/ASi + cMgO/AMgO + (1-cO-cSi-cMgO)/AFe)
    cbarO  = cO  / (AO  /Abar)
    cbarSi = cSi / (ASi /Abar)
    cbarMgO= cMgO/ (AMgO/Abar)    
    return cbarO, cbarSi, cbarMgO

def logKd_ideal(a, b, c, Tcmb, P=135): 
    return a + (b/Tcmb) + (c*P/Tcmb)

def ln_gamma_Fe(X,eps,v):
    
    T1,T2, T3, T4, T5 = 0.0,0.0,0.0,0.0,0.0
     
    i = 0
    for xi in X:
        if xi == 0 : 
            i = i + 1
            continue
        T1 = T1 + eps[i,i] * ( xi + np.log(1.0-xi) )
        i = i + 1
            
    for i in range(0,len(X)-1,1):
        if X[i] == 0 : continue
        for j in range(i+1,len(X),1):
            if X[j] == 0 : continue
            T2 = T2 + eps[i,j] * X[i]    * X[j]    * \
                ( 1.0 + np.log(1.0-X[i])/X[i] + np.log(1-X[j])/X[j] )
            T4 = T4 + eps[i,j] * X[i]**2 * X[j]**2 * \
                ( 1.0/(1-X[i]) + 1.0/(1-X[j]) - 1.0 )
        
    i = 0
    for xi in X:
        j = 0
        if xi == 0 : 
            i = i + 1
            continue        
        for xj in X:
            if xj == 0 or i==j:  
                j = j + 1
                continue
            T3 = T3 + eps[i,j] * xi      * xj      * \
                ( 1.0 + np.log(1-xj)/xj - 1.0/(1.0-xi) )
            T5 = T5 + eps[i,j] * X[i]**2 * X[j]**2 * \
                ( 1.0/(1.0-xj) + 1.0/(1.0-xi) + xi/(2.0*(1.0-xi)**2) -1.0 ) 

            j = j + 1

        i = i + 1

    if v == 1:
        print('gammaFe: T1 = {:10.5e} T2 = {:10.5e} T3 = {:10.5e} T4 = {:10.5e} T5 = {:10.5e} '
            .format(T1, T2, T3, T4, T5))
            
    return T1 - T2 + T3 + 0.5*T4 - T5

def ln_gamma_i(X, eps, lngammaFe, lngammai0, v):
    
    T1,T2,T3 = np.zeros(len(X)),np.zeros(len(X)),np.zeros(len(X))
    tmp2 = np.zeros([len(X), len(X)])
    tmp3 = np.zeros([len(X), len(X)])
    
    i = 0
    for xi in X:   
        if xi == 0 : 
            i = i + 1
            continue     
        T1[i] = eps[i,i] * np.log(1.0-X[i])    
        i = i + 1
    
    i = 0
    for xi in X:
        j = 0
        #if xi == 0 : 
        #    i = i + 1
        #    continue     
        for xj in X:
            if xj == 0 or i==j:  
                j = j + 1
                continue
            T2[i] = T2[i] + eps[i,j] * xj * \
                    ( 1.0 + np.log(1-xj)/xj - 1.0/(1-X[i]) )
            T3[i] = T3[i] + eps[i,j] * X[i] * xj**2 * \
                ( 1.0/(1-X[i]) + 1.0/(1-xj) + X[i]/(2*(1-X[i])**2) - 1.0)
            tmp2[i,j] = eps[i,j] * xj * ( 1.0 + np.log(1-xj)/xj - 1.0/(1-X[i]) )
            tmp3[i,j] = eps[i,j] * X[i] * xj**2 * ( 1.0/(1-X[i]) + 1.0/(1-xj) + X[i]/(2*(1-X[i])**2) - 1.0)
            j = j + 1
        
        if v == 1:
            print('gammai for el {:d}  T1 = {:10.5e} T2 = {:10.5e} T3 = {:10.5e} '
              .format(i, T1[i], T2[i], T3[i]))
            
        i = i + 1
 
    if v == 1:
        print('Sum2 = ', tmp2)
        print('Sum3 = ', tmp3)
            
    return lngammaFe + lngammai0 - T1 - T2 + T3

def min_conc_closure():

    params = {}

    def min_conc(XMg, XO, XSi, XMgO_m, eps, lngamma0, logKd_MgO_B18, T):
            
        XO_tot = XO #+ XMg
        X_C    = np.array( [0.0 , XO_tot, XSi  , 0.0   , XMg] )

        verb = 0
    
        lngammaFe  = ln_gamma_Fe(X_C,eps,verb)                       # eqn S3
        lngammai   = ln_gamma_i( X_C,eps, lngammaFe, lngamma0, verb) # eqn S4
        
        loggammaO  = lngammai[1]/2.303
        loggammaMg = lngammai[4]/2.303

        XMg_ni  = Mg_dissolution(XMgO_m, 10**logKd_MgO_B18,loggammaO,loggammaMg)
        
        if verb == 1:
            # JBs eqn - gives same ans
            print( np.exp ( logKd_MgO_B18*2.303 + np.log(XMgO_m) - lngammai[1] - lngammai[4] ))
            print('T = {:6.1f} gammaC  = {:6.4f} gamma_O  = {:6.4f} gammaSi  = {:6.4f} gammaS  = {:6.4f} gammaMg  = {:6.4f} gammaFe  = {:5.3f}'
           .format(T,np.exp(lngammai[0]),np.exp(lngammai[1]),np.exp(lngammai[2]), p.exp(lngammai[3]),np.exp(lngammai[4]), np.exp(lngammaFe)))
        
        params['lngamma']   = lngammai
        params['lngammaFe'] = lngammaFe
        params['X']         = np.array( [0.0   , XO_tot   , XSi   , 0.0   , XMg_ni] )
                
        return XMg_ni - XMg
    
    return params, min_conc

def O_dissociation(XFe, XO, KO):
    """
        Returns the molar concentration of FeO in the silicate based on an dissociation reaction
    """
    return (XO*XFe) / (KO)

def Si_dissociation(XSi, XO, KSi):
    """
        Returns the molar concentration of Si)2 in the silicate based on an dissociation reaction
    """
    return (XSi*XO**2) / KSi

def Si_exchange(XSi, XFe, XFeO, KSi):
    """
        Returns the molar concentration of Si in the metal based on an exchange reaction
    """
    return (XSi/KSi) * (XFeO/XFe)**2

def Mg_exchange(XMgO, XFe, XFeO, KMg, loggammaO, loggammaMg):
    """B18 eqn 10
        Returns the molar concentration of Mg in the metal based on an exchange reaction
    """
    return 10**( np.log10(KMg) + np.log10(XFe*XMgO) - np.log10(XFeO) - loggammaO - loggammaMg)


def Mg_dissolution(XMgO, KMg, loggammaO, loggammaMg):
    """B18 eqn 9
        Returns the molar concentration of MgO in the metal based on a dissolution reaction
    """
    return 10**( np.log10(KMg) + np.log10(XMgO) - loggammaO - loggammaMg)


def Mg_dissolution_B16(XMgO,KMg):
    """B16 eqn 4
        Returns the molar concentratino of Mg in the metal based on a dissolution reaction
    """
    return 10**( (np.log10(KMg) + np.log10(XMgO))/2.0 )


def Mg_dissociation(XMgO, XO, KMg, loggammaO, loggammaMg):
    """B18 eqn 8
        Returns the molar concentration of Mg in the metal based on a dissociation reaction
    """
    return 10**( np.log(XMgO) + np.log(KMg) - np.log(XO) - loggammaO - loggammaMg)