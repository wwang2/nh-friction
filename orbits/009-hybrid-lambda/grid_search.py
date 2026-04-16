#!/usr/bin/env python3
"""Grid search for Pade friction parameters."""
import math, sys
import numpy as np

M,KT,Q,DT=1.,1.,1.,0.01; NB=5000; NM=200000; TH=10; NS=NM//TH; NT=NB+NM; TC=25000; CS=6.0
W={'h':.024,'d':.2944,'g':.6816}

def gh(q): return q.copy()
def gd(q): x=q[0]; return np.array([4*x*(x*x-1),q[1]])
def gg(q):
    MU=np.array([[3*math.cos(2*math.pi*k/5),3*math.sin(2*math.pi*k/5)] for k in range(5)])
    df=q[None,:]-MU; lw=-0.5*np.sum(df**2,axis=1); lw-=lw.max(); w=np.exp(lw); w/=w.sum()
    return np.einsum('k,kd->d',w,df)

def nc(x): return 0.5*(1+np.array([math.erf(v/1.4142135623730951) for v in x]))
def kl(s,t):
    be=np.linspace(-6,6,101); ct,_=np.histogram(np.clip(s,-6,6),bins=be)
    if ct.sum()==0: return 9
    pe=ct/ct.sum(); lo,hi=be[:-1],be[1:]
    if t=='h': pr=nc(hi)-nc(lo)
    elif t=='d': m=.5*(lo+hi); lp=-(m**2-1)**2; lp-=lp.max(); pr=np.exp(lp)*(hi[0]-lo[0])
    else:
        MX=[3*math.cos(2*math.pi*k/5) for k in range(5)]
        pr=sum((nc(hi-mu)-nc(lo-mu))/5 for mu in MX)
    pr=np.clip(pr,1e-300,None); pr/=pr.sum()
    return sum(p*np.log(p/q) for p,q in zip(pe,pr) if p>0)

def st(x):
    x=np.asarray(x,dtype=np.float64); N=len(x); xc=x-x.mean(); v=float(np.var(xc))
    if v==0: return TC
    nf=1
    while nf<2*N: nf<<=1
    F=np.fft.rfft(xc,n=nf); ac=np.fft.irfft(F*np.conj(F),n=nf)[:N].real; ac=ac/(N-np.arange(N))
    C0=float(ac[0])
    if C0<=0: return TC
    rho=ac/C0; tr=1.0
    for t in range(1,N):
        tr+=2*rho[t]
        if tr<1: tr=1
        if t>CS*tr: return max(tr,1)
    return TC

def r1(a,b,c,gf,dim,seed):
    def ff(xi): xi2=xi*xi; return xi*(a+b*xi2)/(1+c*xi2)
    rng=np.random.default_rng(seed); q=rng.standard_normal(dim); p=rng.standard_normal(dim); xi=0.
    hdt=DT/2; dk=float(dim)*KT; s=np.empty(NS); r=0
    for step in range(NT):
        p=p-gf(q)*hdt; gxi=ff(xi)
        try: ef=math.exp(-gxi*hdt/Q)
        except: return None
        p=p*ef; q=q+p/M*DT; p=p*ef; gq=gf(q); p=p-gq*hdt
        pp=float(np.dot(p,p)); xi=xi+(pp-dk)/Q*DT
        if not(math.isfinite(xi) and np.all(np.isfinite(q)) and np.all(np.isfinite(p))): return None
        if step>=NB and (step-NB)%TH==0: s[r]=q[0]; r+=1
    return s if r==NS else None

def ev(a,b,c):
    tp={}
    for pn,dim,gf,kn in [('h',1,gh,'h'),('d',2,gd,'d'),('g',2,gg,'g')]:
        ts=[]
        for seed in [42,137]:
            s=r1(a,b,c,gf,dim,seed)
            if s is None: return 1e6,{}
            k=kl(s,kn)
            if k>.05: return 1e6,{}
            ts.append(st(s))
        tp[pn]=np.mean(ts)
    return W['h']*tp['h']+W['d']*tp['d']+W['g']*tp['g'], tp

res=[]
n=0
for a in [.64,.66,.68,.70,.72,.74]:
    for b in [2.7,2.8,2.9,3.0,3.1,3.2,3.3]:
        for c in [.04,.05,.06,.07,.08,.09,.10,.11]:
            n+=1
            m,tp=ev(a,b,c)
            if m<1e5:
                res.append((m,a,b,c,tp))
                sys.stdout.write(f'{n}: a={a:.2f} b={b:.1f} c={c:.2f} m={m:.2f}\n')
                sys.stdout.flush()

res.sort()
print('\n=== TOP RESULTS ===')
for m,a,b,c,tp in res[:15]:
    print(f'a={a:.2f} b={b:.1f} c={c:.2f} m={m:.2f} h={tp["h"]:.1f} dw={tp["d"]:.1f} gm={tp["g"]:.1f}')
print(f'Passing: {len(res)}/{n}')
